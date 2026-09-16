"""WP `rs-search-a`（計画 §20.5）の掃引出力を機械的に集計する（判定は書かない）。

入力: `<出力ディレクトリ>/<条件>/<シナリオ>/r3_s<seed>.frames.json`
      （`tests/scripts/rs_scenario_play.py play` が書いたもの）
出力: 条件ごとの集計 JSON（標準出力・`--out` で保存）と、報告に貼る Markdown 表（`--md`）。

集計する量（指示書の「見るもの」）:
  root_visited_pct   … seat 側 kind=main の決定で「訪問された根の手（n>0）の割合」の平均
  q_beats_n          … 同じ決定で「訪問数最多の手より Q が高いのに N で負けた**訪問済み**の手」の件数
                       （n>=1 の手だけ数える＝訪問 0 の手の Q=0 を数えない）
  q_beats_n_dec      … 上が 1 件以上あった決定の数 / seat 側 kind=main の決定数
  rule_moved         … 出した手が「訪問数最多の手」と違った決定の数（q_min_n の効き）
  event_play         … seat 側で選ばれた手のうち「イベントカードの PLAY」の回数
  opp_char_attack    … seat 側で選ばれた ATTACK のうち「相手リーダー以外（＝相手キャラ）」の回数
  winner / final_v   … 9 本の勝敗と seat 側の最終 V（最後に V が出た seat 側の決定の value）
  ms_per_decide      … 1 決定あたりの decide 実測時間（seat/相手・全 kind の中央値と平均・kind=main 別）

`chosen` の同一性は `describe_move`（card_id 基準）で見る。ATTACK の `targets` は card_id なので、
相手リーダーの card_id と突き合わせて「顔か・キャラか」を判定できる。
"""
import argparse
import glob
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_ROOT, _TESTS_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CARDS_PATH = os.path.join(_ROOT, "opcg_sim", "data", "opcg_cards.json")

#: 条件の並び（表の列順）。
COND_ORDER = ["S1", "S2", "S3", "S4", "S5", "R1", "R2"]


def _card_kinds() -> dict:
    with open(CARDS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {c["number"]: (c.get("種類") or "") for c in raw if c.get("number")}


def _seat_of(payload: dict, scenario_seat: str) -> str:
    return scenario_seat


def _argmax_n(stats: list) -> int:
    """訪問数最多の添字（同点は添字が小さい方＝numpy／Rust の argmax 規約）。"""
    best = 0
    for i, s in enumerate(stats):
        if (s.get("n") or 0) > (stats[best].get("n") or 0):
            best = i
    return best


def _rule_moved(d: dict) -> bool:
    """「出した手が訪問数最多の手と違う」か（`candidates` と `value` で見る）。

    `candidates` は等価手マージ後のグループを**訪問数降順**で並べたもの＝`candidates[0]` が
    訪問数最多のグループ、`value` が出したグループの Q（どちらも 3 桁丸め）。よって
    `value != candidates[0]["q"]` なら訪問数最多とは違うグループを出したことになる
    （既定 visits では必ず一致する＝この量は 0 になる。0 でなければ集計の前提が壊れている）。
    """
    cands = d.get("candidates") or []
    v = d.get("value")
    if not cands or v is None or cands[0].get("q") is None:
        return False
    return abs(float(cands[0]["q"]) - float(v)) > 5e-4


def _agg_one(path: str, kinds: dict) -> dict:
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    leaders = (res.get("replay") or {}).get("leaders") or {}
    frames = res.get("frames") or []
    winner = (frames[-1] if frames else {}).get("winner")
    # seat（シナリオの主役席）は最初の決定の player＝分岐点の手番（`scenario["seat"]`）。
    decisions = res.get("decisions") or []
    seat = decisions[0]["player"] if decisions else None
    opp = "p2" if seat == "p1" else "p1"
    opp_leader = leaders.get(opp)

    out = {
        "path": path, "winner": winner, "seat": seat,
        "n_dec": len(decisions), "n_dec_seat": 0, "n_main_seat": 0,
        "visited_frac": [], "visited_pool": [0, 0], "q_beats_n": 0, "q_beats_n_dec": 0, "rule_moved": 0,
        "event_play": 0, "opp_char_attack": 0, "face_attack": 0,
        "final_v": None, "ms_all": [], "ms_main": [],
    }
    for d in decisions:
        ms = d.get("decide_ms")
        if ms is not None:
            out["ms_all"].append(float(ms))
            if d.get("kind") == "main":
                out["ms_main"].append(float(ms))
        if d.get("player") != seat:
            continue
        out["n_dec_seat"] += 1
        ch = d.get("chosen") or {}
        at = ch.get("action_type")
        card = ch.get("card")
        if at == "PLAY" and kinds.get(card) == "イベント":
            out["event_play"] += 1
        if at == "ATTACK":
            tgt = (ch.get("targets") or [None])[0]
            if tgt is not None and tgt == opp_leader:
                out["face_attack"] += 1
            elif tgt is not None:
                out["opp_char_attack"] += 1
        if d.get("value") is not None:
            out["final_v"] = float(d["value"])
        if d.get("kind") != "main":
            continue
        stats = d.get("legal_stats") or []
        if not stats:
            continue
        out["n_main_seat"] += 1
        visited = sum(1 for s in stats if (s.get("n") or 0) > 0)
        out["visited_frac"].append(visited / len(stats))
        b = _argmax_n(stats)
        qb = stats[b].get("q") or 0.0
        nb = stats[b].get("n") or 0
        beats = [i for i, s in enumerate(stats)
                 if (s.get("n") or 0) >= 1 and (s.get("q") or 0.0) > qb and (s.get("n") or 0) < nb]
        out["q_beats_n"] += len(beats)
        out["q_beats_n_dec"] += 1 if beats else 0
        if _rule_moved(d):
            out["rule_moved"] += 1
        out["visited_pool"][0] += visited
        out["visited_pool"][1] += len(stats)
    return out


#: S4／S5 だけで回した 3 本（`--only-key` はこの 3 本に絞る＝条件をまたいで同じ母集団で比べる）。
KEY_SCENARIOS = ("enel_human_20260810_t9-9", "human_enel_vs_roger_20260904_t7-8",
                 "human_doflamingo_vs_luffy_20260904_t10-11")


def aggregate(root: str, only: tuple = ()) -> dict:
    """`only` を渡すとそのシナリオだけで集計する（S4／S5 は 3 本しか回していないので、
    sims 掃引を同じ母集団で比べたいときは `only=KEY_SCENARIOS` にする）。"""
    kinds = _card_kinds()
    conds = {}
    for cond_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(cond_dir):
            continue
        cond = os.path.basename(cond_dir)
        runs = []
        for path in sorted(glob.glob(os.path.join(cond_dir, "*", "r3_s*.frames.json"))):
            if only and os.path.basename(os.path.dirname(path)) not in only:
                continue
            runs.append(_agg_one(path, kinds))
        if not runs:
            continue
        vf = [x for r in runs for x in r["visited_frac"]]
        ms_all = [x for r in runs for x in r["ms_all"]]
        mt = [os.path.getmtime(r["path"]) for r in runs]
        ms_main = [x for r in runs for x in r["ms_main"]]
        conds[cond] = {
            "runs": len(runs),
            "scenarios": sorted({os.path.basename(os.path.dirname(r["path"])) for r in runs}),
            "main_decisions_seat": sum(r["n_main_seat"] for r in runs),
            "decisions_seat": sum(r["n_dec_seat"] for r in runs),
            "decisions_total": sum(r["n_dec"] for r in runs),
            "root_visited_pct": round(100.0 * statistics.fmean(vf), 2) if vf else None,
            "root_visited_pct_pooled": (
                round(100.0 * sum(r["visited_pool"][0] for r in runs)
                      / sum(r["visited_pool"][1] for r in runs), 2)
                if sum(r["visited_pool"][1] for r in runs) else None),
            "q_beats_n": sum(r["q_beats_n"] for r in runs),
            "q_beats_n_dec": sum(r["q_beats_n_dec"] for r in runs),
            "rule_moved": sum(r["rule_moved"] for r in runs),
            "event_play": sum(r["event_play"] for r in runs),
            "opp_char_attack": sum(r["opp_char_attack"] for r in runs),
            "face_attack": sum(r["face_attack"] for r in runs),
            "winners": {os.path.basename(os.path.dirname(r["path"])) + "/" +
                        os.path.basename(r["path"]).split(".")[0]:
                        {"winner": r["winner"], "seat": r["seat"], "final_v": r["final_v"]}
                        for r in runs},
            "decide_seconds_total": round(sum(ms_all) / 1000.0, 1),
            # 壁時計の目安（この条件の最初と最後の出力ファイルの更新時刻の差＋1 本ぶんは含まない）。
            # 掃引は 1 プロセス＝1 シナリオで `JOBS` 並列に回すので、`decide_seconds_total` /
            # 並列度 に近い値になる。
            "wall_seconds_span": round(max(mt) - min(mt), 1) if len(mt) > 1 else None,
            "ms_per_decide_all_median": round(statistics.median(ms_all), 1) if ms_all else None,
            "ms_per_decide_main_median": round(statistics.median(ms_main), 1) if ms_main else None,
            "ms_per_decide_main_mean": round(statistics.fmean(ms_main), 1) if ms_main else None,
            "ms_per_decide_main_max": round(max(ms_main), 1) if ms_main else None,
        }
    return conds


def _order(conds: dict) -> list:
    known = [c for c in COND_ORDER if c in conds]
    return known + sorted(c for c in conds if c not in COND_ORDER)


def to_markdown(conds: dict) -> str:
    cols = _order(conds)
    rows = [
        ("回した本数（シナリオ×seed）", lambda c: c["runs"]),
        ("seat 側 kind=main の決定数", lambda c: c["main_decisions_seat"]),
        ("訪問された根の手の割合（決定ごとの平均 %）", lambda c: c["root_visited_pct"]),
        ("同（手を全部プールした %）", lambda c: c["root_visited_pct_pooled"]),
        ("Q が高いのに N で負けた手（件）", lambda c: c["q_beats_n"]),
        ("同（その手がある決定 / 決定数）",
         lambda c: f"{c['q_beats_n_dec']}/{c['main_decisions_seat']}"),
        ("訪問数最多と違う手を出した決定", lambda c: c["rule_moved"]),
        ("イベント PLAY（seat 側・回）", lambda c: c["event_play"]),
        ("相手キャラ攻撃（seat 側・回）", lambda c: c["opp_char_attack"]),
        ("顔攻撃（seat 側・回）", lambda c: c["face_attack"]),
        ("decide 中央値 ms（kind=main）", lambda c: c["ms_per_decide_main_median"]),
        ("decide 平均 ms（kind=main）", lambda c: c["ms_per_decide_main_mean"]),
        ("decide 最大 ms（kind=main）", lambda c: c["ms_per_decide_main_max"]),
        ("decide 中央値 ms（全 kind）", lambda c: c["ms_per_decide_all_median"]),
        ("decide の合計秒（全 run・全 kind）", lambda c: c["decide_seconds_total"]),
        ("壁時計の幅 秒（4 並列）", lambda c: c["wall_seconds_span"]),
    ]
    out = ["| 指標 | " + " | ".join(cols) + " |",
           "|---|" + "---|" * len(cols)]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(str(fn(conds[c])) for c in cols) + " |")
    return "\n".join(out)


def winners_markdown(conds: dict) -> str:
    """行＝シナリオ/seed・列＝条件・セル＝`winner(seat 側の最終 V)`（`-`＝この条件では回していない）。

    `winner` が `null` の run は「end_turn の TURN_END まで打って決着しなかった」＝勝敗なし。
    """
    cols = _order(conds)
    keys = sorted({k for c in cols for k in conds[c]["winners"]})
    out = ["| シナリオ/seed | seat | " + " | ".join(cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for k in keys:
        seat = next((conds[c]["winners"][k]["seat"] for c in cols if k in conds[c]["winners"]), "?")
        cells = []
        for c in cols:
            w = conds[c]["winners"].get(k)
            if not w:
                cells.append("-")
                continue
            win = w["winner"] if w["winner"] is not None else "決着せず"
            cells.append(f"{win} (V={w['final_v']})")
        out.append(f"| {k} | {seat} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="掃引の出力ディレクトリ（例 scenario_out_a）")
    ap.add_argument("--out", default=None, help="集計 JSON の保存先")
    ap.add_argument("--md", default=None, help="Markdown 表の保存先")
    ap.add_argument("--md-winners", default=None, help="勝敗表（Markdown）の保存先")
    ap.add_argument("--only-key", action="store_true",
                   help="S4／S5 と同じ 3 シナリオだけで集計する（母集団を揃える）")
    args = ap.parse_args(argv)
    conds = aggregate(args.root, KEY_SCENARIOS if args.only_key else ())
    text = json.dumps(conds, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    if args.md_winners:
        with open(args.md_winners, "w", encoding="utf-8") as f:
            f.write(winners_markdown(conds) + "\n")
    md = to_markdown(conds)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
