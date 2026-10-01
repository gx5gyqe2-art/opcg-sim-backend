"""**P8-7(c)**: T123（`transition_ledger.py`）が「攻撃の行の残差 37.8%／38.9%」を
「相手の窓の答え」と「価格の誤り」の混合と読み、**P8 が開くまでこれが最良の分離**と注記した箇所を、
`aux_def`（P8・dump v5）で割る（2026-09-25・読み取り専用・新定数ゼロ）。

## 問い

`transition_ledger.py` の `_priority` は、同じターンの中で**攻撃の型の手の直後**に起きる残差
（`same_turn`・`fam0=="attack"`）を「手の型」で 3 等分した 1 つ（実 37.8%／合成 38.9%）として出す。
コード中の注記（L223-225）どおり、この残差は**相手が窓で応答した分**（ブロック・カウンター——攻撃の
価格には入っていない）と**価格そのものの誤り**（出す・効果の行と同じ性質の誤差）が混ざっている。
`aux_def` で「守り手がその窓で実際に何をしたか」が読めるので、**応答が在った窓と無かった窓に分けて**
残差の大きさを比べる——**応答が無かった窓の残差は、定義上「相手の窓の答え」では説明できない**。

## 式（新定数ゼロ・分けるだけ）

1. `transition_ledger.collect(dirs, dump=dump)` に `dump` を渡し、`same_turn`・`fam0=="attack"` の
   遷移を **局×席×ターン**（攻め手の手番 `(seed, w, t)`）ごとに束ねる——**残差の和だけでなく、束ねた
   遷移の本数も数える**（1 ターンに攻撃が複数回あれば遷移も複数本になる）。
2. `t139_defender_check.defender_actuals_by_turn` で同じ `(seed, w, t)` の**守り手の実際の応答**
   （`blocks`・`counters_used`——`aux_def_row` の row-level 集計。`aux_def` の枠ごとの合計は 6 枠の
   上限で undercount するため使わない・P8-7(b) の限界と同じ）を読む。
3. 応答が在った窓（`blocks>0` または `counters_used>0`）と無かった窓に 2 分し、**遷移 1 本あたりの
   平均残差**（ターンの残差の和 ÷ そのターンの遷移本数）を比べる——**ターンの残差の和のまま比べると
   攻撃の回数が多いターンほど和が大きくなる交絡がある**（実測: 応答した窓は平均 2.59 本／応答が無い
   窓は 1.92 本・35% の差）ので、**本数で割った後の量を主とする**。単位を無理に揃えて 1 つの式には
   しない（観察）。

使い方:

    OPCG_LOG_SILENT=1 python tests/scripts/attack_resid_defender_split.py \
      --in <n_records dump v5>... [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import t139_defender_check as TC  # noqa: E402
import transition_ledger as TL  # noqa: E402


def attack_resid_by_turn(dump):
    """`transition_ledger.collect(dump=...)` の出力を `(seed, w, t)` ごとの
    `{"resid_abs": |resid| の合計, "n": 束ねた遷移の本数}` へ束ねる。"""
    out = {}
    for row in dump:
        key = (row["seed"], row["w"], row["t"])
        e = out.setdefault(key, {"resid_abs": 0.0, "n": 0})
        e["resid_abs"] += row["resid_abs"]
        e["n"] += 1
    return out


def collect(dirs, limit_games=0):
    dump = []
    tl_out = TL.collect(dirs, limit_games, dump=dump)
    resid_by_turn = attack_resid_by_turn(dump)
    defender_actual = TC.defender_actuals_by_turn(dirs, limit_games)

    matched, unmatched = 0, 0
    responded, no_response = [], []      # 各要素: (per_turn_sum, per_transition_mean, n, actual)
    for key, e in resid_by_turn.items():
        actual = defender_actual.get(key)
        if actual is None:
            unmatched += 1
            continue
        matched += 1
        per_turn = e["resid_abs"] / e["n"]
        did_respond = (actual["blocks"] > 0) or (actual["counters_used"] > 0)
        row = (e["resid_abs"], per_turn, e["n"], actual)
        (responded if did_respond else no_response).append(row)

    def _mean(xs):
        return round(float(np.mean(xs)), 6) if xs else None

    def _corr(xs, ys):
        if len(xs) < 2:
            return None
        a, b = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        return round(float(np.corrcoef(a, b)[0, 1]), 4) if a.std() > 0 and b.std() > 0 else None

    resp_sum = [r[0] for r in responded]; noresp_sum = [r[0] for r in no_response]
    resp_pt = [r[1] for r in responded]; noresp_pt = [r[1] for r in no_response]
    resp_n = [r[2] for r in responded]; noresp_n = [r[2] for r in no_response]
    resp_mag = [r[3]["counters_used"] + r[3]["blocks"] for r in responded]

    return {
        "games": tl_out["games"],
        "attack_rows_share_of_total_resid": tl_out.get("resid_priority", {}).get("attack_rows"),
        "n_turns_with_attack_resid": len(resid_by_turn),
        "n_matched": matched, "n_unmatched_no_aux_def": unmatched,
        "n_responded": len(responded), "n_no_response": len(no_response),
        # **交絡の検算**（応答した窓は攻撃の遷移が多いだけかもしれない）
        "n_transitions_mean_responded": _mean(resp_n),
        "n_transitions_mean_no_response": _mean(noresp_n),
        # **素朴な比較**（ターンの残差の和・交絡込み）
        "naive_turn_sum_mean_responded": _mean(resp_sum),
        "naive_turn_sum_mean_no_response": _mean(noresp_sum),
        # **本命**（遷移 1 本あたりに割った後の平均・交絡を除いた比較）
        "resid_per_transition_mean_responded": _mean(resp_pt),
        "resid_per_transition_mean_no_response": _mean(noresp_pt),
        "corr_response_magnitude_vs_resid_per_transition": _corr(resp_mag, resp_pt),
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = collect(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
