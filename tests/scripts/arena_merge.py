"""分散アリーナの台帳マージ（v1・`arena_resume.py` のシャード実行を1判定にまとめる）。

なぜ要るか（2026-09-01 ユーザ決定「アリーナの分散化」）: 1セッションで回せるのは
24ペア×2条件＝192局程度で、この母数では**世代交代の実力差（実測 +0.04＝約+29 Elo）が
CI に埋もれて判定できない**——c9 vs c8 は4本とも 0.521〜0.563 に収まったのに、どの1本も
「0.55 以上かつ CI下限>0.50」を満たせなかった。生成波と同じくオーケストレータで
セッションを分散し、**シャードごとに別 seed 帯**で回した台帳をここで合算する。

判定規約は `arena_resume.final_result` と同一（ペア水準95%CI・promoted は wr≥frac かつ
CI下限>0.50・void は母数から外して件数を必ず載せる）＝集計規約を二重化しない。

**seed 衝突は黙って畳まない**: 同じ seed が複数シャードに現れたら帯設計のミス（同じ対局を
二重計上すると CI が不当に狭まる）なので、重複を数えて明示し、既定では判定を出さずに落とす。

実行例:
  PYTHONPATH=tests python tests/scripts/arena_merge.py --in "/home/user/arena_c9/*/random_*.jsonl"
"""
import argparse
import glob
import json
import os
import sys as _sys

import os as _os  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401


def read_ledger(path):
    """台帳 jsonl → {seed: score|None}（score=None は void）。壊れた行は落とす＝黙って欠測にしない。"""
    done = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sc = r.get("score")
            done[int(r["seed"])] = None if sc is None else float(sc)
    return done


def merge_ledgers(paths):
    """複数台帳を合流（pure）。返り値 (merged, dups, per_file)。

    dups は複数ファイルに現れた seed の一覧＝帯設計の衝突。合流結果からは除外しない
    （呼び出し側が判定を出すか決める）。
    """
    merged, owner, dups, per_file = {}, {}, [], []
    for p in paths:
        d = read_ledger(p)
        per_file.append((p, d))
        for s, sc in d.items():
            if s in owner and owner[s] != p:
                dups.append(s)
            owner[s] = p
            merged[s] = sc
    return merged, sorted(set(dups)), per_file


def summarize(done, frac=0.55):
    """合流台帳 → 判定（`arena_resume.final_result` と同規約・pure）。有効ペアが無ければ None。"""
    from arena_parallel import _pair_level_ci
    valid = [s for s in done if done[s] is not None]
    if not valid:
        return None
    ci = _pair_level_ci([done[s] / 2.0 for s in valid])
    return {"pairs": len(valid), "games": 2 * len(valid),
            "void": len(done) - len(valid),
            "wins": sum(done[s] for s in valid),
            "wr": round(ci["win_rate"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "elo": round(ci["elo"], 1), "elo95": [round(ci["elo_lo"], 1), round(ci["elo_hi"], 1)],
            "promoted": bool(ci["win_rate"] >= frac and ci["lo"] > 0.50)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="台帳 jsonl（glob 可・シャードぶん並べる）")
    ap.add_argument("--frac", type=float, default=0.55)
    ap.add_argument("--label", default="", help="出力に付ける見出し（条件名など）")
    ap.add_argument("--allow-dup-seeds", action="store_true",
                    help="seed 衝突があっても判定を出す（既定は落とす＝二重計上を隠さない）")
    args = ap.parse_args()

    paths = []
    for pat in args.src:
        paths += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("台帳が1つも見つからない", flush=True)
        return 2

    merged, dups, per_file = merge_ledgers(paths)
    head = f"[{args.label}] " if args.label else ""
    for p, d in per_file:
        r = summarize(d, args.frac)
        if r:
            print(f"  {os.path.basename(p):28s} {r['wins']:.1f}/{r['games']} = {r['wr']:.4f}"
                  f" void {r['void']}", flush=True)
    if dups:
        print(f"⚠ seed 衝突 {len(dups)} 件（帯設計のミス＝同じ対局の二重計上）: "
              f"{dups[:8]}{'...' if len(dups) > 8 else ''}", flush=True)
        if not args.allow_dup_seeds:
            print("判定は出さない（--allow-dup-seeds で強制可）", flush=True)
            return 1
    res = summarize(merged, args.frac)
    if res is None:
        print("有効ペアが無い（全 void）＝判定を出さない", flush=True)
        return 1
    res["shards"] = len(paths)
    res["dup_seeds"] = len(dups)
    print(f"{head}ARENA_MERGE_FINAL " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["promoted"] else 1


if __name__ == "__main__":
    _sys.exit(main())
