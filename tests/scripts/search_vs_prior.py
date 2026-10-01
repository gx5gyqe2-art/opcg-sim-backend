"""探索は π（生成役の事前分布）を超えているか——同じ局面の `pol_p` と訪問分布を比べる。

v14 の dump は方策点ごとに **探索前の P**（`pol_p`＝生成役ネットの根の出力）と **探索後の訪問分布**
（`pi`＝正規化した訪問回数）と各手の Q を持つ。π の教師は後者なので、両者がほぼ同じなら π は
自分の写しを学んでいるだけ（環が回っていない）。両者が離れていれば探索が先読みで何かを足している。

出すもの（全体・候補数 k の帯・ターン帯・forced の有無で層別）:
  - top1_agree  … argmax(pi) == argmax(pol_p) の割合（探索が答えを変えなかった割合）
  - tv_raw      … 0.5·Σ|pi − pol_p|（全変動距離）
  - tv_temp     … 木が実際に出発した事前分布（pol_p^(1/T)・T=--root-prior-temp）との全変動距離
  - H_pi / H_p  … 訪問分布と事前分布のエントロピー（探索が尖らせたか・平らにしたか）
  - q_backs_override … 答えを変えた点のうち Q(argmax pi) > Q(argmax p) の割合
                       （PUCT は Q の高い方へ訪問を寄せるので高くて当然＝弱い指標。参考値）
  - chosen_is_top1  … 実際に打った手が argmax(pi) だった割合（temp_turns・ε の影響の目安）

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/search_vs_prior.py \\
    --in ~/n32_wave/w01/n_records/* ~/n32_wave/w02/n_records/* \\
    --root-prior-temp 2.0 --out ~/svp_w32.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.learned.train import dump_io as DIO  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402


def _ent(x):
    x = np.clip(np.asarray(x, np.float64), 1e-12, None)
    x = x / x.sum()
    return float(-(x * np.log(x)).sum())


def _k_band(k):
    return "k2" if k <= 2 else "k3-4" if k <= 4 else "k5-8" if k <= 8 else "k9+"


def _turn_band(t):
    return "t1-4" if t <= 4 else "t5-8" if t <= 8 else "t9+"


def _agg(rows):
    if not rows:
        return {"points": 0}
    a = {k: np.array([r[k] for r in rows], np.float64) for k in
         ("agree", "tv_raw", "tv_temp", "h_pi", "h_p", "chosen_top1")}
    ov = [r["q_gain"] for r in rows if r["q_gain"] is not None]
    return {"points": len(rows),
            "top1_agree": float(a["agree"].mean()),
            "tv_raw": float(a["tv_raw"].mean()),
            "tv_temp": float(a["tv_temp"].mean()),
            "H_pi": float(a["h_pi"].mean()),
            "H_p": float(a["h_p"].mean()),
            "chosen_is_top1": float(a["chosen_top1"].mean()),
            "overrides": len(ov),
            "q_backs_override": (float(np.mean([g > 0 for g in ov])) if ov else None),
            "q_gain_mean": (float(np.mean(ov)) if ov else None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="dump v4 のディレクトリ（pol_p 必須）")
    ap.add_argument("--holdout-mod", type=int, default=0, help="0＝全方策点・N＝seed%%N==0 だけ")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 点だけ（0＝全部）")
    ap.add_argument("--root-prior-temp", type=float, default=2.0,
                    help="生成時の root_prior_temp（木が出発した事前分布＝pol_p^(1/T)）")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    _s, _ab, _abm, _pwr, _isl, vocab = build_eff_tables()
    V, P, C = DIO.load_dump(args.src, vocab, with_policy=True, cache_dir=args.cache_dir, n_tok=NL.N_TOK)
    if C.get("p") is None:
        raise SystemExit("この dump には pol_p が無い（v14 の波だけが対象）")
    L = np.asarray(P["len"]); ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    seed = np.asarray(P["seed"]); chosen = np.asarray(P["chosen"]); row = np.asarray(P["row"])
    pi_all = np.asarray(C["pi"], np.float32); p_all = np.asarray(C["p"], np.float32)
    q_all = np.asarray(C["q"], np.float32)
    turn = np.asarray(V["turn"])[row]
    forced = DIO.load_row_col(args.src, "forced")
    forced_pt = (np.asarray(forced)[row] if forced is not None else None)

    pts = np.arange(len(L))
    if args.holdout_mod:
        pts = pts[seed[pts] % args.holdout_mod == 0]
    if args.limit:
        pts = pts[:args.limit]
    print(f"方策点 {len(pts)}（全 {len(L)}・{time.time()-t0:.0f}s）", flush=True)

    inv_t = 1.0 / max(args.root_prior_temp, 1e-6)
    rows = []
    for i in pts:
        k = int(L[i])
        if k < 2:
            continue
        sl = slice(ptr[i], ptr[i] + k)
        pi = pi_all[sl].astype(np.float64); p = p_all[sl].astype(np.float64); q = q_all[sl]
        if pi.sum() <= 0 or p.sum() <= 0:
            continue
        pi = pi / pi.sum(); p = p / p.sum()
        pt = np.power(np.clip(p, 1e-12, None), inv_t); pt = pt / pt.sum()
        a_pi = int(np.argmax(pi)); a_p = int(np.argmax(p))
        ch = int(chosen[i])
        rows.append({
            "k": k, "turn": int(turn[i]),
            "forced": (int(forced_pt[i]) if forced_pt is not None else 0),
            "agree": float(a_pi == a_p),
            "tv_raw": float(0.5 * np.abs(pi - p).sum()),
            "tv_temp": float(0.5 * np.abs(pi - pt).sum()),
            "h_pi": _ent(pi), "h_p": _ent(p),
            "chosen_top1": float(0 <= ch < k and ch == a_pi),
            "q_gain": (float(q[a_pi] - q[a_p]) if a_pi != a_p else None),
        })

    out = {"src": [os.path.abspath(s) for s in args.src], "root_prior_temp": args.root_prior_temp,
           "holdout_mod": args.holdout_mod, "all": _agg(rows), "by_k": {}, "by_turn": {}, "by_forced": {}}
    for band in ("k2", "k3-4", "k5-8", "k9+"):
        out["by_k"][band] = _agg([r for r in rows if _k_band(r["k"]) == band])
    for band in ("t1-4", "t5-8", "t9+"):
        out["by_turn"][band] = _agg([r for r in rows if _turn_band(r["turn"]) == band])
    for f in (0, 1, 2, 3):
        sub = [r for r in rows if r["forced"] == f]
        if sub:
            out["by_forced"][f"forced{f}"] = _agg(sub)

    def line(name, a):
        if not a.get("points"):
            return f"  {name:8} (0 点)"
        qb = a["q_backs_override"]
        return (f"  {name:8} n {a['points']:6d}  agree {a['top1_agree']:.3f}  tv_raw {a['tv_raw']:.3f}"
                f"  tv_temp {a['tv_temp']:.3f}  H_pi {a['H_pi']:.3f}  H_p {a['H_p']:.3f}"
                f"  chosen=top1 {a['chosen_is_top1']:.3f}"
                + (f"  q_backs {qb:.3f} (n {a['overrides']})" if qb is not None else ""))

    print(line("all", out["all"]))
    for sec in ("by_k", "by_turn", "by_forced"):
        for name, a in out[sec].items():
            print(line(name, a))
    print(f"  {time.time()-t0:.0f}s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
