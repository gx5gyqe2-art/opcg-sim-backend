# 合成 w39 の深い探索の写し（syn_resolution §1.2）: theory_map.py playdeep の点の組 (デッキ, 帯) を (全部, 帯) にまとめ直して
# `--per`（帯ごとの点の数）で選ぶ薄い包み。選び方（局と step のハッシュの小さい順）・打ち直し・探索は playdeep そのまま。1 回きり・テストなし。
import collections, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tests", "scripts"))
import theory_map_new as TM
_orig = TM.playgap_points
def _by_band(src, rowlist=None):
    pts, cnt, dof = _orig(src, rowlist)
    out = collections.defaultdict(list)
    for (deck, band), xs in pts.items():
        out[("all", band)].extend(xs)
    return out, cnt, dof
TM.playgap_points = _by_band
if __name__ == "__main__":
    TM.main(sys.argv[1:]) if hasattr(TM, "main") else None
