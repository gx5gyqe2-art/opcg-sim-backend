"""**C-5**: 攻撃の遷移で**自分の耐久(Θ_me)がなぜ動くのか**を、耐久の3つの項(ライフ・手札・体)と
攻め手の種類(リーダー／ブロッカー／ブロッカーでない体)で割る(2026-09-25・読み取り専用・新定数ゼロ)。

## 問い

C-1は攻撃の型の残差が`th_me`(負)と`th_opp`(正)の2軸に乗ると読んだが、**その`sh`は席0視点**
(`transition_ledger._swap_state`)で、席1が攻めた行では軸が入れ替わり符号も逆になる——「攻め手の
耐久」として読めていなかった。本器は`attack_detail`(攻め手視点へ写した`sh_att`と両席のΘの3項)を
結果×攻め手の種類で束ね、**Θ_meのどの項が実際に動いたか**を直接数える。

## 式(分けるだけ)

* `d_me` … `Θ_me(行1) − Θ_me(行0)` の3項(同じターンなので手札1枚の価格`g`は行0で固定)
* `d_opp` … 同じく相手の耐久の3項
* `unpriced_opp` … `Σd_opp + v`(価格`v`は`Θ_opp`を`−v`動かす想定なので、0なら価格どおり)
* `moved` … 手札／ドン／ライフ／アクティブなブロッカー数が行0→行1で変わった行の割合

使い方:

    OPCG_LOG_SILENT=1 python tests/scripts/attack_theta_parts.py --in <n_records>... [--games N] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kappa_vector as KV  # noqa: E402
import transition_ledger as TL  # noqa: E402

PARTS = ("life", "hand", "body")
KINDS = ("leader", "blocker", "nonblocker")


def kind_of(mv):
    """攻め手の種類: リーダー／ブロッカー（体・イベントでない）／それ以外の体。"""
    if mv.get("leader"):
        return "leader"
    return "blocker" if mv.get("blocker") else "nonblocker"


def _mean(xs):
    return round(float(np.mean(xs)), 6) if xs else None


def _group(rows):
    d_me = np.array([[r["me1"][i] - r["me0"][i] for i in range(3)] for r in rows], float)
    d_opp = np.array([[r["opp1"][i] - r["opp0"][i] for i in range(3)] for r in rows], float)
    vs = [r["mv"].get("v") for r in rows]
    unp = [float(d_opp[k].sum()) + float(v) for k, v in enumerate(vs) if v is not None]
    def moved(key):
        return round(float(np.mean([abs(r["row1"][key] - r["row0"][key]) > 1e-9 for r in rows])), 4)
    return {"n": len(rows),
            "sh_att_th_me": _mean([r["sh_att"]["th_me"] for r in rows]),
            "sh_att_th_opp": _mean([r["sh_att"]["th_opp"] for r in rows]),
            "d_me": {p: round(float(d_me[:, i].mean()), 6) for i, p in enumerate(PARTS)},
            "d_me_total": round(float(d_me.sum(axis=1).mean()), 6),
            "d_opp": {p: round(float(d_opp[:, i].mean()), 6) for i, p in enumerate(PARTS)},
            "d_opp_total": round(float(d_opp.sum(axis=1).mean()), 6),
            "v_mean": _mean([v for v in vs if v is not None]),
            "unpriced_opp": _mean(unp),
            "moved": {"my_hand": moved("my_hand"), "my_don": moved("my_don"), "my_life": moved("my_life"),
                      "n_me_blk": moved("n_me_blk"), "n_me_rest": moved("n_me_rest"),
                      "opp_life": moved("opp_life"), "opp_hand": moved("opp_hand"),
                      "n_opp_chr": moved("n_opp_chr"), "n_opp_blk": moved("n_opp_blk")},
            "src_rest1_share": _mean([1.0 if r["mv"].get("src_rest1") else 0.0 for r in rows]),
            "don_k_mean": _mean([float(r["mv"].get("don_k") or 0) for r in rows])}


def summarize(dump):
    """`dump`(`transition_ledger.collect(dump=)`の攻撃の行)を 結果×攻め手の種類・結果・種類 で束ねる。"""
    by_rk, by_r, by_k = {}, {}, {}
    for row in dump:
        k = kind_of(row["mv"])
        by_rk.setdefault((row["resp"], k), []).append(row)
        by_r.setdefault(row["resp"], []).append(row)
        by_k.setdefault(k, []).append(row)
    n = max(1, len(dump))
    return {"n": len(dump),
            "kind_share": {k: round(len(v) / n, 4) for k, v in by_k.items()},
            "by_kind": {k: _group(v) for k, v in by_k.items()},
            "by_result": {r: _group(v) for r, v in by_r.items()},
            "by_result_kind": {"%s/%s" % (r, k): _group(v) for (r, k), v in sorted(by_rk.items())}}


def collect(dirs, limit_games=0, d_mode=None):
    if d_mode:
        KV.set_d_mode(d_mode)
    dump = []
    TL.collect(dirs, limit_games, dump=dump)
    out = summarize(dump)
    out["d_mode"] = KV.D_MODE
    out["attack_rest_mode"] = KV.ATTACK_REST_MODE
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--d-mode", default=None, choices=KV.D_MODES)
    ap.add_argument("--attack-rest", dest="attack_rest", default=None, choices=KV.ATTACK_REST_MODES)
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.attack_rest:
        KV.set_attack_rest_mode(a.attack_rest)
    out = collect(a.src, a.games, a.d_mode)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
