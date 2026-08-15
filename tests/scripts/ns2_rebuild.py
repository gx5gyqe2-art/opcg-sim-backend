"""ns2 評価行列の再構築（2026-08-14・コンテナロールバック対策で計器化）。

`holdout_ns2` fixture（meta.jsonl 群＝24世界ラベル）から (tag,i,ev) を集め、リプレイ復元→
指定世代で符号化→単一 npz を書く。ラベルは fixture の ev（再ロールアウト不要・数分で完了）。
/tmp の評価行列はコンテナロールバックで頻繁に消えるため、必要時に本器で再構築する。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/ns2_rebuild.py \\
    --enc-version 11 --out /tmp/holdout_ns2_rows_v11c.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
import rl_encoder as E  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXDIR = os.path.join(REPO, "tests", "fixtures", "candidates", "holdout_ns2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metas = []
    for d in sorted(glob.glob(os.path.join(FIXDIR, "part*"))):
        mf = os.path.join(d, "meta.jsonl")
        if os.path.exists(mf):
            metas += [json.loads(l) for l in open(mf)]
    seen, rows = set(), []
    for m in metas:
        k = (m["tag"], m["i"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(m)

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    db = _load_db()
    eng = LearnedEngine()
    cache = {}
    out = {"scalars": [], "field": [], "card_idx": [], "value": [], "tagi": []}
    miss = 0
    for r in rows:
        tag, i = r["tag"], r["i"]
        if tag not in cache:
            raw = RE.load_replay_json(table[tag])
            rec = raw.get("replay", raw)
            cache[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                          rec["actions"])
        rec, fbi, acts = cache[tag]
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str) or built is None:
            miss += 1
            continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        e = E.encode(m0, name, eng.vocab, version=args.enc_version)
        out["scalars"].append(e["scalars"])
        out["field"].append(e["field"])
        out["card_idx"].append(e["card_idx"])
        out["value"].append(np.float32(r["ev"]))
        out["tagi"].append(f"{tag}@{i}")
    L = max(len(c) for c in out["card_idx"])
    ci = np.zeros((len(out["card_idx"]), L), np.int64)
    for k, c in enumerate(out["card_idx"]):
        ci[k, :len(c)] = c
    np.savez_compressed(args.out, scalars=np.stack(out["scalars"]),
                        field=np.stack(out["field"]), card_idx=ci,
                        value=np.array(out["value"], np.float32),
                        tagi=np.array(out["tagi"]))
    print(f"NS2_REBUILD_DONE rows={len(out['value'])} miss={miss}"
          f" enc_version={args.enc_version}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
