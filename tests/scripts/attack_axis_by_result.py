"""**C-1**: 攻撃の値段の残差(`transition_ledger.py`・T123)を、攻撃の結果別・軸別に割る
(2026-09-25・読み取り専用・新定数ゼロ)。

## 問い

`transition_ledger.py`は攻撃の型の同じターン内残差(全体の37.9%/38.4%・基準線report参照)を
出すが、**その残差がどの軸(自分の耐久・相手の耐久・自分の速さ・相手の速さ・ターン番号)に乗るか**、
**攻撃の結果(通った/カウンターされた/ブロッカーが倒れた/何も起きない)で違うか**は測っていなかった。
候補: (a)攻撃した体が自分の耐久を動かす(レストになる・結果を問わず起きる規則) (b)奪ったライフ札が
相手の手札の項を増やす(「通った」行だけで起きる) (c)付けたドンが自分の速さの財布を減らす。

## 式(新定数ゼロ・分けるだけ)

`transition_ledger.collect(dirs, dump=dump)`に`dump`を渡すと、攻撃の型の同じターン内残差の遷移
ごとに、既存の`attack_response.classify`(結果の4分類+複合)と、既存のシャープレイ配分(`sh`・5軸)が
乗る(いずれも既存の関数の呼び出しを追加しただけ・式は変えていない)。本器はこれを結果別に束ねて
平均を出すだけ。

使い方:

    OPCG_LOG_SILENT=1 python tests/scripts/attack_axis_by_result.py --in <n_records>... [--d-mode curve|theory]
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

AXES5 = ("th_me", "th_opp", "a_me", "a_opp", "j")


def by_result(dump):
    """攻撃の型の同じターン内残差の遷移(`dump`)を結果別に束ね、軸ごとの平均(符号つき)と
    `resid_abs`の平均を出す。"""
    by_resp = {}
    for row in dump:
        by_resp.setdefault(row["resp"], []).append(row)
    out = {}
    for resp, rows in by_resp.items():
        out[resp] = {
            "n": len(rows),
            "resid_abs_mean": round(float(np.mean([r["resid_abs"] for r in rows])), 6),
            "axis_mean": {a: round(float(np.mean([r["sh"][a] for r in rows])), 6) for a in AXES5},
        }
    return out


def collect(dirs, limit_games=0, d_mode=None):
    if d_mode:
        KV.set_d_mode(d_mode)
    dump = []
    out = TL.collect(dirs, limit_games, dump=dump)
    return {"games": out["games"], "d_mode": KV.D_MODE,
            "attack_rows_share_of_total_resid": out.get("resid_priority", {}).get("attack_rows"),
            "by_result": by_result(dump)}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--d-mode", default=None, choices=KV.D_MODES)
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = collect(a.src, a.games, a.d_mode)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
