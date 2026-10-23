"""r21 の方策の教材（波 49・50 の再分析の点）と元の記録・訓練の読み込みの対応を全点で照合する（`docs/n_loop_ops.md` §7.38 の 1）。

読み取りだけ（エンジン・ネット・既定の挙動は変えない）。1 回きりの照合の器＝テストは足さない。数える項目:

- (a) 局面の符号化: 再分析の行（`n{W}_rean/w{XX}` の npz）の行の列（`card_idx`・`scalars`・`tokens`・`field`・`sig`・
  `pol_chosen` 等・`pol_v0` を除く全部）が、元の記録（`n_records/n{W}_w{XX}_p*` の同じ seed・同じ step の行）とビットで同じか。
- (b) 候補の並び: 候補の列（`pol_sig`・`pol_cid`・`pol_k`・`pol_p`・`pol_si`・`pol_ti` 等・`pol_n`／`pol_q` を除く全部）が
  元の記録と同じ順で同じか。`points.jsonl` の `rec_n` が記録の `pol_n` と同じか・`pol_n`／`pol_q`／`pol_v0` が `points.jsonl`
  の深い読みの値と同じか。選んだ手（`pol_chosen`）の候補の `pol_sig` が行の `sig` と同じか。
- (c) 教師の最大の手: 訓練と同じ式（`q_improved_pi`・`pol_p`・`pol_v0`）で作った π' の最大の手が、`reanalyze.py` の「深い最善」
  （`points.jsonl` の深い訪問 `n` の最大）と同じか。違う点では π' の最大の手が深い読みで何回訪問されたか（最大の訪問に対する比）。
- (d) 訓練の読み込み: `dump_io.load_dump`（`n_rel_train` が使う口）で読んだ方策点の候補の添字・訪問・値・P・行（盤面）が npz と
  揃っているか。
- (e) 符号化の版: `meta` の `enc_version`・`dump_version`・`pol_v0` の有無・`tokens`／`scalars` の形が訓練側の現行の形と揃うか。
- (f) 訓練側の前向き計算と Rust の一致（任意・`--net`）: 訓練の器（`NRelNet.policy_logits`）で生成役ネットを候補に当てた方策を、
  再分析の深い読みが返した根の事前分布 `p_deep`（同じネット・`root_prior_temp` で丸めた後）と比べる。

    python tests/scripts/r21_teacher_align.py --rean ~/rean --rec ~/rec/n_records --net opcg_sim/data/learned/nrel_r13d.npz \\
        --out n1_results/r21_diag/align.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opcg_sim.learned.train import dump_io as DIO                 # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT              # noqa: E402

ROW_SKIP = {"pol_v0"}
POL_SKIP = {"pol_n", "pol_q"}
ROW_POL = ("pol_len", "pol_chosen", "pol_v0")


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def load_npz_cols(f):
    z = np.load(f, allow_pickle=True)
    cols = {k: z[k] for k in z.files}
    ptr = np.concatenate([[0], np.cumsum(cols["pol_len"].astype(np.int64))])
    pol = [k for k in cols if k.startswith("pol_") and k not in ROW_POL]
    return cols, ptr, pol


def record_index(rec_dirs):
    """(seed, step) → (file, 行)。"""
    idx = {}
    for d in rec_dirs:
        for f in DIO.shard_files(d):
            z = np.load(f, allow_pickle=True)
            for i, (s, st) in enumerate(zip(z["seed"], z["step"])):
                idx[(int(s), int(st))] = (f, i)
    return idx


def check_shard(rean_dir, rec_dirs, c_scale, n_min, c_visit):
    t0 = time.time()
    ridx = record_index(rec_dirs)
    cache = {}
    pts = {}
    for line in open(os.path.join(rean_dir, "points.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        if "game" not in x and not x.get("unmapped"):
            pts[(int(x["seed"]), int(x["step"]))] = x
    cnt = {k: 0 for k in ("points", "row_missing", "a_row_mismatch", "b_pol_mismatch", "b_rec_n_mismatch",
                          "b_deep_n_mismatch", "b_deep_q_mismatch", "b_v0_mismatch", "b_chosen_sig_mismatch",
                          "b_chosen_neg", "points_jsonl_missing", "c_qimp_argmax_ne_deep_best",
                          "c_visits_argmax_ne_deep_best", "c_deep_best_ne_rec_best", "c_qimp_argmax_ne_rec_best",
                          "c_qimp_argmax_lowvisit")}
    bad_cols = {}
    lowvis = []
    cand = []
    for f in DIO.shard_files(rean_dir):
        rc, rptr, rpol = load_npz_cols(f)
        n = len(rc["seed"])
        lens = rc["pol_len"].astype(np.int64)
        pi = NT.q_improved_pi(lens, rc["pol_n"], rc["pol_q"], rc["pol_p"], v0=rc["pol_v0"].astype(np.float64),
                              n_min=n_min, c_visit=c_visit, c_scale=c_scale)
        for i in range(n):
            cnt["points"] += 1
            key = (int(rc["seed"][i]), int(rc["step"][i]))
            if key not in ridx:
                cnt["row_missing"] += 1
                continue
            rf, j = ridx[key]
            if rf not in cache:
                cache.clear()
                cache[rf] = load_npz_cols(rf)
            oc, optr, opol = cache[rf]
            bad = [k for k in rc if k not in rpol and k not in ROW_SKIP and (k not in oc or not eq(rc[k][i], oc[k][j]))]
            if bad:
                cnt["a_row_mismatch"] += 1
                for k in bad:
                    bad_cols[k] = bad_cols.get(k, 0) + 1
            a, b = rptr[i], rptr[i + 1]
            oa, ob = optr[j], optr[j + 1]
            badp = [k for k in rpol if k not in POL_SKIP and (k not in oc or not eq(rc[k][a:b], oc[k][oa:ob]))]
            if badp:
                cnt["b_pol_mismatch"] += 1
                for k in badp:
                    bad_cols[k] = bad_cols.get(k, 0) + 1
            ch = int(rc["pol_chosen"][i])
            if ch < 0:
                cnt["b_chosen_neg"] += 1
            elif json.loads(str(rc["pol_sig"][a + ch])) != json.loads(str(rc["sig"][i])):
                cnt["b_chosen_sig_mismatch"] += 1
            p = pts.get(key)
            if p is None:
                cnt["points_jsonl_missing"] += 1
                continue
            rec_n = oc["pol_n"][oa:ob].astype(np.float64)
            if not np.allclose(np.asarray(p["rec_n"]), rec_n):
                cnt["b_rec_n_mismatch"] += 1
            dn = np.asarray(p["n"], np.float64)
            if not np.allclose(rc["pol_n"][a:b].astype(np.float64), dn.astype(rc["pol_n"].dtype)):
                cnt["b_deep_n_mismatch"] += 1
            if not np.allclose(rc["pol_q"][a:b].astype(np.float64),
                               np.asarray(p["q"]).astype(rc["pol_q"].dtype).astype(np.float64)):
                cnt["b_deep_q_mismatch"] += 1
            if abs(float(rc["pol_v0"][i]) - float(np.float16(p["v0"]) if rc["pol_v0"].dtype == np.float16 else p["v0"])) > 1e-3:
                cnt["b_v0_mismatch"] += 1
            deep_best = int(np.argmax(dn))
            rec_best = int(np.argmax(rec_n))
            qa = int(np.argmax(pi[a:b]))
            va = int(np.argmax(rc["pol_n"][a:b]))
            cnt["c_qimp_argmax_ne_deep_best"] += int(qa != deep_best)
            cnt["c_visits_argmax_ne_deep_best"] += int(va != deep_best)
            cnt["c_deep_best_ne_rec_best"] += int(deep_best != rec_best)
            cnt["c_qimp_argmax_ne_rec_best"] += int(qa != rec_best)
            if qa != deep_best:
                r = float(dn[qa] / max(dn.max(), 1e-9))
                lowvis.append(r)
                cnt["c_qimp_argmax_lowvisit"] += int(r < 0.05)
            cand.append(b - a)
    lv = np.asarray(lowvis)
    return cnt, bad_cols, {
        "qimp_argmax_visit_ratio_when_ne": ({"n": int(len(lv)), "median": float(np.median(lv)),
                                             "p10": float(np.percentile(lv, 10)),
                                             "share_lt_0.05": float(np.mean(lv < 0.05)),
                                             "share_lt_0.20": float(np.mean(lv < 0.20))} if len(lv) else None),
        "cands_mean": float(np.mean(cand)) if cand else None, "sec": round(time.time() - t0, 1)}


def check_loader(rean_dirs, vocab, cache_dir):
    """(d) `dump_io.load_dump` の方策点が npz と揃っているか（候補の添字・n・q・p・pi・盤面の行）。"""
    V, P, C = DIO.load_dump(rean_dirs, vocab, with_policy=True, cache_dir=cache_dir, n_tok=None)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    k = 0
    off_v = 0
    bad = {"n": 0, "q": 0, "p": 0, "pi": 0, "len": 0, "chosen": 0, "v0": 0, "row_tok": 0, "row_sc": 0, "row_ci": 0,
           "at": 0, "si": 0, "ti": 0}
    total = 0
    for d in rean_dirs:
        for f in DIO.shard_files(d):
            rc, rptr, _ = load_npz_cols(f)
            take = np.where((rc["kind"] == 0) & (rc["pol_len"] >= 2) & (rc["pol_chosen"] >= 0))[0]
            for i in take:
                total += 1
                a, b = rptr[i], rptr[i + 1]
                ca, cb = ptr[k], ptr[k + 1]
                bad["len"] += int(int(P["len"][k]) != b - a)
                bad["chosen"] += int(int(P["chosen"][k]) != int(rc["pol_chosen"][i]))
                bad["v0"] += int(abs(float(P["v0"][k]) - float(rc["pol_v0"][i])) > 1e-6)
                bad["n"] += int(not np.allclose(C["n"][ca:cb], rc["pol_n"][a:b].astype(np.float32)))
                bad["q"] += int(not np.allclose(C["q"][ca:cb], rc["pol_q"][a:b].astype(np.float32)))
                bad["p"] += int(not np.allclose(C["p"][ca:cb], rc["pol_p"][a:b].astype(np.float32)))
                nn = rc["pol_n"][a:b].astype(np.float64)
                bad["pi"] += int(not np.allclose(C["pi"][ca:cb], nn / max(nn.sum(), 1e-9), atol=1e-6))
                bad["at"] += int(not np.array_equal(
                    C["at"][ca:cb], [NT.N1._atype_idx(json.loads(str(s))[0]) for s in rc["pol_sig"][a:b]]))
                bad["si"] += int(not np.array_equal(C["si"][ca:cb], rc["pol_si"][a:b]))
                bad["ti"] += int(not np.array_equal(C["ti"][ca:cb], rc["pol_ti"][a:b]))
                row = int(P["row"][k])
                bad["row_tok"] += int(not np.array_equal(
                    np.asarray(V["tok"][row], np.float32)[:rc["tokens"].shape[1], :rc["tokens"].shape[2]],
                    rc["tokens"][i].astype(np.float16).astype(np.float32)))
                bad["row_sc"] += int(not np.array_equal(np.asarray(V["sc"][row], np.float32)[:rc["scalars"].shape[1]],
                                                        rc["scalars"][i].astype(np.float16).astype(np.float32)))
                ci_v = np.asarray(V["ci"][row], np.int64)
                bad["row_ci"] += int(not np.array_equal(ci_v, rc["card_idx"][i][:len(ci_v)].astype(np.int64)))
                k += 1
            off_v += len(rc["seed"])
    assert k == len(P["len"]), (k, len(P["len"]))
    return {"points": total, "loader_points": int(len(P["len"])), "mismatch": bad,
            "val_points_seed_mod7": int(np.sum(P["seed"] % 7 == 0))}


def check_forward(rean_dirs, net_path, cache_dir, temp):
    """(f) 訓練の器の前向き（生成役ネット）と、深い読みが返した Rust の根の事前分布 `p_deep` を比べる。"""
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = DIO.load_dump(rean_dirs, vocab, with_policy=True, cache_dir=cache_dir, n_tok=NL.N_TOK)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    net = NT.NRelNet.load(net_path, tables=tables)
    # p_deep を (seed, step) で引く
    pdeep = {}
    for d in rean_dirs:
        for line in open(os.path.join(d, "points.jsonl"), encoding="utf-8"):
            x = json.loads(line)
            if "game" not in x and not x.get("unmapped") and x.get("p_deep"):
                pdeep[(int(x["seed"]), int(x["step"]))] = np.asarray(x["p_deep"], np.float64)
    keys = []
    for d in rean_dirs:
        for f in DIO.shard_files(d):
            z = np.load(f, allow_pickle=True)
            take = np.where((z["kind"] == 0) & (z["pol_len"] >= 2) & (z["pol_chosen"] >= 0))[0]
            keys += [(int(z["seed"][i]), int(z["step"][i])) for i in take]
    am = l1 = tot = 0
    l1s = []
    bs = 512
    for s in range(0, len(P["len"]), bs):
        bi = np.arange(s, min(s + bs, len(P["len"])))
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        sc, ci, tok = NT.prow(V, P, bi)
        rel_om, rel_oo = NT.relations_or_zeros(net, ci, tok, rt)
        tab = net.card_table()
        lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, C["si"][idx].astype(np.int64),
                               C["ti"][idx].astype(np.int64), net.cand_feats(C, idx, tab), C["budget"][idx], tab=tab)
        p = net.seg_softmax(lo, seg, len(bi))
        pos = 0
        for j, L in enumerate(lens):
            pd = pdeep.get(keys[bi[j]])
            if pd is not None and len(pd) == L:
                pp = np.maximum(p[pos:pos + L].astype(np.float64), 0) ** (1.0 / temp)
                pp /= pp.sum()
                pd = pd / max(pd.sum(), 1e-12)
                tot += 1
                am += int(np.argmax(pp) == np.argmax(pd))
                d1 = float(np.abs(pp - pd).sum())
                l1s.append(d1)
            pos += L
    l1s = np.asarray(l1s)
    return {"net": os.path.basename(net_path), "points": tot, "argmax_agree": am / max(tot, 1),
            "l1_mean": float(l1s.mean()) if tot else None, "l1_median": float(np.median(l1s)) if tot else None,
            "l1_p99": float(np.percentile(l1s, 99)) if tot else None, "share_l1_gt_0.05": float(np.mean(l1s > 0.05)) if tot else None,
            "root_prior_temp": temp}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rean", required=True, help="再分析の置き場（n49_rean/w01 … を持つ dir）")
    ap.add_argument("--rec", required=True, help="元の記録の n_records（n49_w01_p0 … を持つ dir）")
    ap.add_argument("--waves", default="49,50")
    ap.add_argument("--shards", default="01,02,03,04,05,06,07,08")
    ap.add_argument("--net", default=None, help="(f) 生成役ネット（再分析の深い読みを打ったネット）")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"shards": {}, "totals": {}, "bad_cols": {}}
    rean_dirs = []
    for w in a.waves.split(","):
        for xx in a.shards.split(","):
            rd = os.path.join(a.rean, "n%s_rean" % w, "w%s" % xx)
            rean_dirs.append(rd)
            recs = sorted(glob.glob(os.path.join(a.rec, "n%s_w%s_p*" % (w, xx))))
            m = json.load(open(os.path.join(rd, "meta_n_record.json"), encoding="utf-8"))
            m0 = json.load(open(os.path.join(recs[0], "meta_n_record.json"), encoding="utf-8"))
            cnt, bc, extra = check_shard(rd, recs, NT.PI_C_SCALE, NT.PI_N_MIN, NT.PI_C_VISIT)
            z = np.load(DIO.shard_files(rd)[0], allow_pickle=True)
            zr = np.load(DIO.shard_files(recs[0])[0], allow_pickle=True)
            meta = {"enc_version": m.get("enc_version"), "rec_enc_version": m0.get("enc_version"),
                    "dump_version": m.get("dump_version"), "rec_dump_version": m0.get("dump_version"),
                    "has_pol_v0": "pol_v0" in z.files, "rec_has_pol_v0": "pol_v0" in zr.files,
                    "has_pol_p": "pol_p" in z.files, "tokens": list(z["tokens"].shape[1:]),
                    "scalars": int(z["scalars"].shape[1]), "rec_parts": [os.path.basename(r) for r in recs],
                    "search": m.get("search"), "rec_search": m0.get("search"), "reanalyzed": m.get("reanalyzed")}
            out["shards"]["n%s_w%s" % (w, xx)] = {"counts": cnt, "bad_cols": bc, "meta": meta, **extra}
            for k, v in cnt.items():
                out["totals"][k] = out["totals"].get(k, 0) + v
            for k, v in bc.items():
                out["bad_cols"][k] = out["bad_cols"].get(k, 0) + v
            print("ALIGN %s %s" % ("n%s_w%s" % (w, xx), json.dumps(cnt)), flush=True)
    tok_shape, sc_dim = DIO.target_form()
    out["trainer_form"] = {"tokens": list(tok_shape), "scalars": int(sc_dim)}
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    vocab = build_eff_tables()[-1]
    out["loader"] = check_loader(rean_dirs, vocab, a.cache_dir)
    print("LOADER " + json.dumps(out["loader"]), flush=True)
    if a.net:
        temps = {json.dumps((s["meta"]["search"] or {}).get("root_prior_temp")) for s in out["shards"].values()}
        temp = float(json.loads(sorted(temps)[0]) or 1.0)
        out["forward"] = check_forward(rean_dirs, a.net, a.cache_dir, temp)
        print("FORWARD " + json.dumps(out["forward"]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("TOTALS " + json.dumps(out["totals"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
