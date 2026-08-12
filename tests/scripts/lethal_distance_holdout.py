"""リーサル距離Δ特徴の一般60点ホールドアウト検査（v52-v3・2026-08-12）。

v52 スパイクの採用判定手順（`cpu_v52_lethal_distance_spike_20260812.md` §3）:
「忠実度の改善を1段入れてから、同じ58点＋**一般60点ホールドアウト**で再試験」の後半。
乖離族（58点）で効く特徴が、一般盤面で**害を出さない**（ライフ差等の既存信号と矛盾する
ノイズにならない）ことを確認する。

盤面: `tests/fixtures/candidates/v51_teacher/holdout60_boards.json`
（bb1 ホールドアウトと同一の60点＝リプレイ (tag, action_index) とレフェリー勝率ラベル
 sims48×6世界。生成コマンドは同 json の origin に記録）。

出力: v1（無抵抗）/ v2（防御込み）/ v3（忠実度改善）の距離差とライフ差それぞれの
EV への符号一致・r。採用判定の観点＝v3 が一般盤面でもライフ差と少なくとも同等の
説明力を持ち、乖離族と符号が逆になっていないこと。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/lethal_distance_holdout.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=None, help="既定=fixtures の holdout60_boards.json")
    args = ap.parse_args()
    import time
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    import mark_gate as MG
    import replay_reeval as RE
    import coach_gate as CG
    from lethal_distance_probe import lethal_distance

    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = args.boards or os.path.join(REPO, "tests", "fixtures", "candidates",
                                       "v51_teacher", "holdout60_boards.json")
    spec = json.load(open(path))
    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}

    db = _load_db()
    gs = OPCGGame()
    gs_raw = OPCGGame(prune_futile=False)   # v3 用（枝刈りで過剰ドン付与の手が消えないように）

    cache = {}
    rows = []
    t0 = time.time()
    for r in spec["rows"]:
        tag, i = r["tag"], r["i"]
        if tag not in cache:
            raw = RE.load_replay_json(table[tag])
            rec = raw.get("replay", raw)
            cache[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                          rec["actions"])
        rec, fbi, acts = cache[tag]
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str) or built is None:
            print(f"  {tag}@{i}: 復元不可（スキップ）", flush=True)
            continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        opp = m0.p2.name if m0.p1.name == name else m0.p1.name
        d1a = lethal_distance(gs, m0, name)
        d1b = lethal_distance(gs, m0, opp)
        d2a = lethal_distance(gs, m0, name, defend=True)
        d2b = lethal_distance(gs, m0, opp, defend=True)
        d3a = lethal_distance(gs_raw, m0, name, defend=True, v3=True)
        d3b = lethal_distance(gs_raw, m0, opp, defend=True, v3=True)
        me = m0.p1 if m0.p1.name == name else m0.p2
        op_ = m0.p2 if m0.p1.name == name else m0.p1
        rows.append({**r, "d_me": d1a, "d_opp": d1b, "d_me_def": d2a, "d_opp_def": d2b,
                     "d_me_v3": d3a, "d_opp_v3": d3b,
                     "life_diff": len(me.life or []) - len(op_.life or [])})
        print(f"  {tag}@{i} T{r['turn']} {name}: v1=({d1a},{d1b}) v2=({d2a},{d2b})"
              f" v3=({d3a},{d3b}) ev={r['ev']:+.2f} {time.time()-t0:.0f}s", flush=True)

    ev = np.array([r["ev"] for r in rows], float)
    ld = np.array([r["life_diff"] for r in rows], float)
    nz = ev != 0                                    # EV=0（勝率5割）は符号判定から除外
    print(f"\n=== 一般{len(rows)}点ホールドアウト（レフェリー勝率ラベル・EV≠0 は {int(nz.sum())}点）")
    for label, a, b in (("無抵抗v1 ", "d_opp", "d_me"), ("防御込みv2", "d_opp_def", "d_me_def"),
                        ("防御込みv3", "d_opp_v3", "d_me_v3")):
        dd = np.array([r[a] - r[b] for r in rows], float)
        oks = int((np.sign(dd)[nz] == np.sign(ev)[nz]).sum())
        tie = int((dd[nz] == 0).sum())
        r_ = float(np.corrcoef(dd, ev)[0, 1])
        print(f"  {label}: 符号一致 {oks}/{int(nz.sum())}（引分 {tie}）  r={r_:+.3f}")
    okl = int((np.sign(ld)[nz] == np.sign(ev)[nz]).sum())
    print(f"  ライフ差 : 符号一致 {okl}/{int(nz.sum())}（引分 {int((ld[nz]==0).sum())}）"
          f"  r={float(np.corrcoef(ld, ev)[0,1]):+.3f}")
    print("LETHAL_HOLDOUT " + json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
