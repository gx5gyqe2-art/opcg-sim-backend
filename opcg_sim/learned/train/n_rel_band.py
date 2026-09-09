"""n_rel_band: 評価帯（dump v2／v3／v4 の holdout 行）で N系 c ネットと NRel r ネットの value を
同じ行で比べる（2026-09-05・r1 の判定用）。

**問い**: 訓練 val（`n_rel_train.py` の ep 行）は r1 自身の数字しか出ない。**同じ holdout 行**で
既定 c10 の v_mse/v_sign を出し、行を揃えて比べる。dump v2 の scalars は v13＝v12（94 列）の
末尾に 29 列を足した append-only なので、c ネットには先頭 94 列と card_idx を渡せばよい。

**補助教師（dump v4・2026-09-10・計画 §20.8.2）**: dump に `aux` 列があり、ネットが補助ヘッドを
持っていれば **aux の各列の予測誤差**（Huber／BCE・`n_rel_train.aux_losses` と同じ式）も出す。
さらに**層別**で読む——`deck_kinds` 列（WP `rs-removal-decks`）があればその型・色で、無ければ
**リーダー色**（seed から `decks.leader_pair` で引き直す＝dump に列が無くても復元できる）で
層を作り、加えて「**除去を打った直後の行**」（選んだ手が PLAY／ACTIVATE_MAIN で、その主体
カードが除去系）の層を持つ。各層で z 予測誤差と aux 誤差を並べる。

実行例:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_band \\
    --in ~/n23_wave/w01/n23_records ~/n23_wave/w02/n23_records \\
    --nrel ~/nrel_r3.npz ~/nrel_r3_aux.npz --out /tmp/band.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import sys as _sys

import numpy as np


from opcg_sim.learned.train import dump_io as DIO  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402
from opcg_sim.learned import n_eff as NE  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned import n_rel_feat as NR  # noqa: E402
from opcg_sim.learned.encoder import SCALARS_V12  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402  aux の列名（正本）


def _load(dirs, mod, limit, cache_dir=None):
    """holdout 行（`seed%mod==0`）だけを float32 で持つ（読みは `dump_io.load_dump`＝memmap）。

    帯は holdout（既定 1/7）しか使わないので、**そこだけ切り出して RAM に置く**。dump v2／v3 の
    どちらでも同じ値になる（pack が float16／int16 の正本）。`keep` は**全行に対する index**＝
    npz の行ごとの列（`sig`／`deck_kinds`）を後から同じ順で引くために返す。"""
    V, _P, _C = DIO.load_dump(dirs, {}, with_policy=False, cache_dir=cache_dir,
                              n_tok=NE.MAX_CI)     # c ネットは 24 枠・r ネットは先頭 22 枠を使う
    keep = np.where(V["seed"] % mod == 0)[0]
    if limit:
        keep = keep[:limit]
    D = {"sc": np.asarray(V["sc"][keep], np.float32),
         "ci": np.asarray(V["ci"][keep], np.int64),
         "tok": np.asarray(V["tok"][keep], np.float32),
         "z": np.asarray(V["z"][keep], np.float32),
         "seed": np.asarray(V["seed"][keep], np.int64),
         "turn": np.asarray(V["turn"][keep], np.int16)}
    if V.get("aux") is not None:
        D["aux"] = np.asarray(V["aux"][keep], np.float32)
        D["aux_tok"] = np.asarray(V["aux_tok"][keep], np.float32)
        D["aux_mask"] = np.asarray(V["aux_mask"][keep], np.float32)
    return D, keep


def _metrics(v, z):
    return {"v_mse": float(np.mean((v - z) ** 2)), "v_sign": float(np.mean((v > 0) == (z > 0))),
            "n": int(len(z))}


def aux_errors(pred, pred_tok, aux, aux_tok, mask):
    """aux の**列ごと**の誤差（0〜9 は Huber・aux_tok は [BCE, Huber, BCE] を枠平均）。"""
    m = mask > 0
    if not m.any():
        return None
    out = {n: float(NT.huber(pred[m, j] - aux[m, j]).mean())
           for j, n in enumerate(RG.AUX_COLS)}
    pt, at = pred_tok[m], aux_tok[m]
    out["tok_attacked"] = float(NT.bce_logits(pt[:, :, 0], at[:, :, 0]).mean())
    out["tok_life_pushed"] = float(NT.huber(pt[:, :, 1] - at[:, :, 1]).mean())
    out["tok_ability"] = float(NT.bce_logits(pt[:, :, 2], at[:, :, 2]).mean())
    out["n"] = int(m.sum())
    return out


# ---------------------------------------------------------------------------
# 層別（deck_kinds があればそれ・無ければリーダー色／除去を打った直後の行）
# ---------------------------------------------------------------------------
def _leader_colors(db, seeds, who):
    """行ごとの「自分のリーダーの色」（`decks.leader_pair` を seed から引き直す）。"""
    from opcg_sim.loop import decks as D
    cache = {}
    out = []
    for s, w in zip(seeds.tolist(), who.tolist()):
        pair = cache.get(s)
        if pair is None:
            pair = cache[s] = D.leader_pair(db, int(s), "random")
        cid = pair[0 if int(w) == 0 else 1]
        card = db.get_card(cid) if cid else None
        cols = sorted({getattr(x, "name", str(x)) for x in (getattr(card, "colors", ()) or ())})
        out.append("|".join(cols) if cols else "?")
    return np.array(out)


def _removal_rows(db, card_ids, sig, ats=("PLAY", "ACTIVATE_MAIN")):
    """「除去を打った直後の行」のマスク（選んだ手が PLAY／ACTIVATE_MAIN で主体が除去系）。

    分類は `deck_roles.classify`（WP `rs-removal-decks`）があればそれを使い、無ければ
    `n_rel_feat.roles_of` の removal／reduce／lock のどれかが立つカードを除去系とする。
    """
    try:                                              # 並行 WP が入っていれば使う
        from opcg_sim.loop import deck_roles          # noqa: F401
        classify = deck_roles.classify
    except Exception:                                 # noqa: BLE001  無ければ roles_of で代用
        classify = None
    ok = np.zeros(len(card_ids), bool)
    hit = {}
    for i, cid in enumerate(card_ids):
        if not cid:
            continue
        at = None
        try:
            at = json.loads(sig[i])[0] if sig is not None else None
        except Exception:                             # noqa: BLE001  壊れた行は層に入れない
            continue
        if sig is not None and at not in ats:
            continue
        r = hit.get(cid)
        if r is None:
            card = db.get_card(cid)
            if card is None:
                r = hit[cid] = False
            elif classify is not None:
                r = hit[cid] = bool(classify(card))
            else:
                v = NR.roles_of(card)
                r = hit[cid] = bool(v[0] or v[1] or v[2])
        ok[i] = r
    return ok


def build_buckets(D, db, dirs, keep):
    """層のリスト `[(name, mask)]`。`all` ＋ ターン帯 ＋（deck_kinds ／ リーダー色）＋ 除去直後。"""
    n = len(D["z"])
    buckets = [("all", np.ones(n, bool))]
    if D["turn"].any():
        buckets += [(f"turn{lo}-{hi}", (D["turn"] >= lo) & (D["turn"] <= hi))
                    for lo, hi in ((1, 4), (5, 8), (9, 99))]
    who = DIO.load_row_col(dirs, "who")
    kinds = DIO.load_row_col(dirs, "deck_kinds")       # WP rs-removal-decks（無ければ None）
    if kinds is not None:
        lay = [json.loads(s) if s else {} for s in np.asarray(kinds)[keep].astype(str)]
        for key in ("injected", "colors"):
            vals = sorted({v for d in lay for v in (d.get(key) or ())})
            for v in vals:
                m = np.array([v in (d.get(key) or ()) for d in lay])
                if m.sum() >= 32:
                    buckets.append((f"{key}:{v}", m))
    elif who is not None:
        col = _leader_colors(db, D["seed"], np.asarray(who)[keep])
        for v in sorted(set(col.tolist())):
            m = col == v
            if m.sum() >= 32:
                buckets.append((f"leader_color:{v}", m))
    cids = DIO.chosen_card_ids(dirs)
    sig = DIO.load_row_col(dirs, "sig")
    if cids is not None:
        m = _removal_rows(db, np.asarray(cids)[keep].astype(str),
                          np.asarray(sig)[keep].astype(str) if sig is not None else None)
        if m.any():
            buckets.append(("after_removal", m))
            buckets.append(("not_removal", ~m))
    return buckets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="dump v2／v3／v4 のディレクトリ")
    ap.add_argument("--neff", nargs="*", default=[], help="N系 c ネット npz（複数可）")
    ap.add_argument("--nrel", nargs="*", default=[], help="NRel r ネット npz（複数可）")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 行だけ（0＝全 holdout 行）")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    ap.add_argument("--cache-dir", default=None,
                    help="dump の pack（float16/int16 の .npy・§18.5）の置き場所"
                         f"（既定は ${DIO.CACHE_ENV} か ~/.cache/opcg/dump_pack）")
    ap.add_argument("--zero-rel", action="store_true",
                    help="serve 時の遮断: 関係 R（rel_om/rel_oo）を 0 にして評価（r ネットのみ）")
    ap.add_argument("--zero-opp-pool", action="store_true",
                    help="serve 時の遮断: 相手デッキ知識（EXTRA の opp_pool_* 列）を 0 にして評価")
    args = ap.parse_args(argv)

    t0 = time.time()
    from opcg_sim.learned.vocab import load_db   # 退避で cpu_selfplay が消えたため（2026-09-07）
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    D, keep = _load(args.src, args.holdout_mod, args.limit, args.cache_dir)
    z = D["z"].astype(np.float64)
    print(f"holdout 行 {len(z)}（seed%{args.holdout_mod}==0・{time.time()-t0:.0f}s）", flush=True)
    preds = {}
    aux_pred = {}
    for p in args.neff:
        net = NE.NEffNet.load(p, tables=tables)
        sc = D["sc"][:, :SCALARS_V12]
        preds[os.path.basename(p)] = np.concatenate(
            [net.value(sc[s:s + args.bs], D["ci"][s:s + args.bs]) for s in range(0, len(z), args.bs)])
    if args.nrel:
        ptab = NR.profile_table(db, vocab)
        rt = NR.RelTable(ptab)
    sc_r = D["sc"]
    if args.zero_opp_pool:
        sc_r = sc_r.copy()
        for j, name in enumerate(NR.EXTRA_COLS):
            if name.startswith("opp_pool_"):
                sc_r[:, SCALARS_V12 + j] = 0.0
    for p in args.nrel:
        net = NL.NRelNet.load(p, tables=tables)
        want_aux = bool(net.aux and "aux" in D)
        vs, pa, pt = [], [], []
        for s in range(0, len(z), args.bs):
            ci = D["ci"][s:s + args.bs][:, :NL.N_TOK]; tok = D["tok"][s:s + args.bs]
            rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
            if args.zero_rel:
                rel_om = np.zeros_like(rel_om); rel_oo = np.zeros_like(rel_oo)
            if want_aux:
                v, a, at = net.value_with_aux(sc_r[s:s + args.bs], ci, tok, rel_om, rel_oo)
                pa.append(a); pt.append(at)
            else:
                v = net.value(sc_r[s:s + args.bs], ci, tok, rel_om, rel_oo)
            vs.append(v)
        tag = os.path.basename(p) + ("" if not args.zero_rel else "+zero_rel") \
            + ("" if not args.zero_opp_pool else "+zero_opp_pool")
        preds[tag] = np.concatenate(vs)
        if want_aux:
            aux_pred[tag] = (np.concatenate(pa), np.concatenate(pt))
    buckets = build_buckets(D, db, args.src, keep)
    res = {}
    for name, v in preds.items():
        res[name] = {b: _metrics(v[m], z[m]) for b, m in buckets if m.any()}
        ap_ = aux_pred.get(name)
        if ap_ is not None:
            for b, m in buckets:
                if not m.any():
                    continue
                e = aux_errors(ap_[0][m], ap_[1][m], D["aux"][m], D["aux_tok"][m],
                               D["aux_mask"][m])
                if e:
                    res[name][b]["aux"] = e
        print(f"  {name}")
        for b, r in res[name].items():
            line = f"    {b:28s} mse {r['v_mse']:.4f} sign {r['v_sign']:.3f} n {r['n']:6d}"
            if "aux" in r:
                a = r["aux"]
                line += (f" | aux {np.mean([a[c] for c in RG.AUX_COLS]):.4f}"
                         f" life {a['opp_turn_my_life_lost']:.3f}"
                         f" atk {a['opp_turn_attacks']:.3f}"
                         f" tok_atk {a['tok_attacked']:.3f} tok_ab {a['tok_ability']:.3f}"
                         f" (n {a['n']})")
            print(line, flush=True)
    print(f"  {time.time()-t0:.0f}s")
    out = {"rows": int(len(z)), "src": args.src, "buckets": [b for b, m in buckets if m.any()],
           "nets": res}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")
        # `--out` があるときの N_REL_BAND は**要約**（層別を全部載せると 1 行が数万字になる）。
        print("N_REL_BAND " + json.dumps(
            {"rows": out["rows"], "src": args.src, "out": args.out, "buckets": len(out["buckets"]),
             "nets": {k: v.get("all") for k, v in res.items()}}, ensure_ascii=False))
    else:
        print("N_REL_BAND " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
