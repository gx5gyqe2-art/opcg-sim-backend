"""WP `rs-search-a` の `RESULT.json` を集計から組む（§20.5 の成果物の形に合わせるだけ）。

使い方:
  python tests/scripts/rs_search_a_result.py scenario_out_a \
      --make-test "N passed" --notes "..." --latency latency.json --out RESULT.json

`--latency` は「1 決定あたりの decide 時間を**単独プロセスで**測った結果」
（`{"<sims>": {"median_ms": .., "mean_ms": .., "max_ms": .., "decisions": N}}`）。掃引そのものは
4 並列で回すので、そのままではレイテンシの上限の目安にならない。
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import rs_search_a_agg as agg  # noqa: E402
import rs_search_a_keys as keys  # noqa: E402

#: 条件 → sims（レイテンシの引き当てと報告の見出しに使う）。
COND_SIMS = {"S1": 160, "S2": 640, "S3": 2560, "S4": 10240, "S5": 16000,
             "R1": 160, "R2": 640}


def build(root: str, make_test: str, notes: str, latency: dict) -> dict:
    a = agg.aggregate(root)
    k = keys.keys_for(root)
    conds = agg._order(a)
    key_names = ["enel_t9_kaminosabaki_before_attack", "enel_roger_t7_gammaknife",
                 "doffy_t10_root_visited", "doffy_t10_root_legal", "doffy_t10_winner"]
    out = {
        "job": "rs-search-a",
        "status": "done",
        "conditions": conds,
        "condition_settings": {
            c: {"sims": COND_SIMS.get(c),
                "select_rule": "q_min_n" if c.startswith("R") else "visits",
                "q_min_frac": 0.125 if c.startswith("R") else None,
                "root_prior_temp": 2.0 if c.startswith("R") else 1.0,
                "scenarios": len(a[c]["scenarios"]), "runs": a[c]["runs"]}
            for c in conds},
        "key_decisions": {name: {c: k.get(c, {}).get(name) for c in conds} for name in key_names},
        "key_decision_detail": {
            "enel_t9_kami_vs_oom": {c: k.get(c, {}).get("detail_enel_t9_kami_vs_oom") for c in conds},
            "enel_roger_t7_gamma": {c: k.get(c, {}).get("detail_enel_roger_t7_gamma") for c in conds},
            "doffy_t10_first_main": {c: k.get(c, {}).get("detail_doffy_t10_first_main")
                                     for c in conds},
        },
        "q_beats_n_count": {c: a[c]["q_beats_n"] for c in conds},
        "q_beats_n_decisions": {c: f"{a[c]['q_beats_n_dec']}/{a[c]['main_decisions_seat']}"
                                for c in conds},
        "root_visited_pct": {c: a[c]["root_visited_pct"] for c in conds},
        "root_visited_pct_pooled": {c: a[c]["root_visited_pct_pooled"] for c in conds},
        "rule_moved": {c: a[c]["rule_moved"] for c in conds},
        "event_play": {c: a[c]["event_play"] for c in conds},
        "opp_char_attack": {c: a[c]["opp_char_attack"] for c in conds},
        "face_attack": {c: a[c]["face_attack"] for c in conds},
        "winners": {c: a[c]["winners"] for c in conds},
        "latency_ms_per_decide": latency,
        "latency_ms_per_decide_sweep_4jobs": {
            c: {"main_median": a[c]["ms_per_decide_main_median"],
                "main_mean": a[c]["ms_per_decide_main_mean"],
                "main_max": a[c]["ms_per_decide_main_max"],
                "all_median": a[c]["ms_per_decide_all_median"]} for c in conds},
        "decide_seconds_total": {c: a[c]["decide_seconds_total"] for c in conds},
        "make_test": make_test,
        "notes": notes,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--make-test", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--latency", default=None)
    ap.add_argument("--status", default="done")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    latency = {}
    if args.latency:
        with open(args.latency, encoding="utf-8") as f:
            latency = json.load(f)
    data = build(args.root, args.make_test, args.notes, latency)
    data["status"] = args.status
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[result] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
