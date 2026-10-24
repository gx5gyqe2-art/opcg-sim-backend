"""守りの再監査の修正後の 300 局を組む（WP rs-window-tree-blocker・1 回きり）。

手が分かれなかった局（`diverge_<B>.jsonl` の diverged=false）は修正前の局と同じ（決定的・勝者と手番数で 1 局ずつ照合済み）
なので、修正前の監査の記録（点と確かめ）をそのまま使う。分かれた局は
  - 全部を打ち直した局（`defense_<B>/points_seeds_div_seeds_<B>.jsonl`・最初に回した分）はその記録、
  - 残りは、最初に手が分かれた点より前は修正前の記録・その点から先は `defense_partial.py` の記録
を使う。

  python merge_defense.py <B> points   # → defense_<B>/merged/points_<B>_300.jsonl と new_points.jsonl（確かめる点）
  （コミットしたのは gzip した points_<B>_300.jsonl.gz だけ・new_points は打ち直した 2 つのファイルの連結）
  python merge_defense.py <B> verify   # → defense_<B>/merged/verify.jsonl（修正前の確かめ＋new_points の確かめ）
"""
import gzip
import json
import os
import sys

D = "n1_results/window_tree_blocker"
BEFORE = {3013000: ("n1_results/window_read/defense_new/points_3013000_300.jsonl.gz",
                    "n1_results/window_read/defense_new/verify.jsonl"),
          3351000: ("n1_results/reaudit_wr/defense/points_3351000_300.jsonl",
                    "n1_results/reaudit_wr/defense/verify.jsonl")}


def lines(p):
    if not os.path.exists(p):
        return []
    fh = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
    return [json.loads(x) for x in fh]


def seed_of(r):
    return r["game"]["seed"] if "game" in r else r["seed"]


def complete(rows):
    done = {r["game"]["seed"] for r in rows if "game" in r}
    return [r for r in rows if seed_of(r) in done], done


def main():
    b, stage = int(sys.argv[1]), sys.argv[2]
    div = {}
    for d in map(json.loads, open(f"{D}/diverge_{b}.jsonl")):
        if d["diverged"]:
            div[d["seed"]] = min(p["step"] for p in d["probes"] if not p["same"])
    bp, bv = BEFORE[b]
    full, full_seeds = complete(lines(f"{D}/defense_{b}/points_seeds_div_seeds_{b}.jsonl"))
    part, part_seeds = complete(lines(f"{D}/defense_{b}_partial/points_seeds_part_seeds_{b}.jsonl"))
    assert full_seeds | part_seeds == set(div) and not (full_seeds & part_seeds), "打ち直した局が分かれた局と一致しない"
    before = lines(bp)
    keep_before = lambda r: seed_of(r) not in div or (seed_of(r) in part_seeds and "game" not in r
                                                       and r["step"] < div[seed_of(r)])
    out = f"{D}/defense_{b}/merged"
    os.makedirs(out, exist_ok=True)
    if stage == "points":
        pts = [r for r in before if keep_before(r)] + full + part
        new = full + part
        for name, rows in ((f"points_{b}_300.jsonl", pts), ("new_points.jsonl", new)):
            with open(f"{out}/{name}", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(b, "games", sum("game" in r for r in pts), "diverged", len(div), "full", len(full_seeds),
              "partial", len(part_seeds), "points", len(pts), "new", len(new))
    else:
        ver = [r for r in lines(bv) if keep_before(r)] + lines(f"{out}/new_verify/verify.jsonl")
        with open(f"{out}/verify.jsonl", "w", encoding="utf-8") as f:
            for r in ver:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(b, "verify rows", len(ver))


if __name__ == "__main__":
    main()
