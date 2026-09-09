"""WP `rs-search-a`（§20.5）の「1 決定あたりの decide 時間」を**同じ局面**で測る。

シナリオの分岐点（`start_turn` の開始盤面）を復元し、**同じ 1 局面**に対して sims だけを変えて
`decide` を呼び、実測時間を出す。`rs_scenario_play.py play` を sims 別に回して比べると、
sims が変われば打つ手も変わる＝決定の集合そのものが違うので、レイテンシの比較にならない
（同じ局面を測るのがこの計器の存在理由）。

serve に載せられる上限の目安を読むための計器なので、**単独プロセスで・他に負荷が無い状態で**
回す。1 局面 1 sims につき `--repeat` 回（既定 1）測って中位を採る。

使い方:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_search_a_latency.py \
      --scenarios enel_human_20260810_t9-9 human_enel_vs_roger_20260904_t7-8 \
                  human_doflamingo_vs_luffy_20260904_t10-11 \
      --sims 160 640 2560 10240 16000 --out latency.json

`--worlds K`（§20.7.1）を足すと 1 決定で世界を K 本引いて木を並列に回す＝**同じ sims の
まま**「K 本ぶんの実効 sims」になる。K を変えて回せば並列化の壁時計コストが読める
（sims と違い、K は 1 決定あたりの読みの本数を変える）。
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_ROOT, _TESTS_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: E402,F401

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

import rs_scenario_play as scp  # noqa: E402


def measure(name: str, sims_list: list, seed: int, repeat: int, opts: dict) -> dict:
    scenario = scp._load_scenario(name)
    payload = scp._load_json_maybe_gz(scp._resolve_replay_source(scenario["source"]))
    start_idx = hb.turn_start_index(payload, scenario["start_turn"])
    hidden = hb.frame_to_hidden(payload, start_idx, seed)
    load_engine()
    out = {}
    for sims in sims_list:
        samples = []
        legal = None
        for _ in range(repeat):
            # `rs_scenario_play._play_one` と同じ基点＝同じ世界サンプルを見る。
            random.seed(f"{name}:{seed}")
            game = RsGame.from_hidden(hidden, seed=seed)
            pending = game.get_pending_request()
            tr: dict = {}
            t0 = time.perf_counter()
            game.decide(pending["player_id"], trace=tr, sims=sims, **opts)
            samples.append((time.perf_counter() - t0) * 1000.0)
            legal = len(tr.get("legal_stats") or [])
        out[str(sims)] = {"ms": round(statistics.median(samples), 1),
                          "ms_all": [round(x, 1) for x in samples],
                          "legal": legal, "kind": tr.get("kind")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--sims", nargs="+", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--select-rule", default=None, choices=("visits", "q_min_n"))
    ap.add_argument("--q-min-frac", type=float, default=None)
    ap.add_argument("--root-prior-temp", type=float, default=None)
    ap.add_argument("--worlds", type=int, default=None,
                    help="1 決定あたりの世界サンプル本数（§20.7.1・既定 1）")
    # §20.7.2（WP `rs-setup-box`）: 準備箱。既定（渡さない）＝serve 既定のまま。
    ap.add_argument("--setup-box", dest="setup_box", action="store_true", default=None)
    # §20.7.8 の 5: 共通規則だけの on/off（既定＝--setup-box に従う）。
    ap.add_argument("--select-branch", dest="select_branch", default=None,
                    choices=("on", "off"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    # "on"／"off" → bool（argparse は type の後に choices を見るので変換はここで）。
    select_branch = None if args.select_branch is None else (args.select_branch == "on")
    opts = {"select_rule": args.select_rule, "q_min_frac": args.q_min_frac,
            "root_prior_temp": args.root_prior_temp, "worlds": args.worlds,
            "setup_box": args.setup_box, "select_branch": select_branch}
    data = {"note": "同じ 1 局面（分岐点の開始盤面）に対する 1 decide の実測 ms・単独プロセス",
            "seed": args.seed, "repeat": args.repeat,
            "opts": {k: v for k, v in opts.items() if v is not None},
            "by_scenario": {}}
    for name in args.scenarios:
        data["by_scenario"][name] = measure(name, args.sims, args.seed, args.repeat, opts)
        print(f"[latency] {name}: "
              + " ".join(f"{s}={v['ms']}ms" for s, v in data["by_scenario"][name].items()))
    # sims ごとの中位（シナリオをまたぐ）＝報告の 1 行。
    data["median_ms_by_sims"] = {
        str(s): round(statistics.median(
            [data["by_scenario"][n][str(s)]["ms"] for n in args.scenarios]), 1)
        for s in args.sims}
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[latency] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
