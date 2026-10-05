"""H-4g の測定結果（複数の作業ブランチに分かれて出る json）を 1 つの表にまとめる（`extract_g.py` の後継）。

使い方（土台ブランチの `h4_wip/results/` と、3 つの結果ブランチを取ってから）:

    git fetch origin claude/h4-rule-don-wip claude/h4-real claude/h4-syn claude/h4-misc
    python tests/scripts/h4_tables.py                      # 既定の探し方（下）
    python tests/scripts/h4_tables.py --src DIR --src git:origin/claude/h4-real   # 好きに指定

**探し方**: 各 json を `--src` の順に探し、**最初に見つかったもの**を使う（既定は
`h4_wip/results`〔作業ツリー〕→ `git:origin/claude/h4-real` → `-syn` → `-misc` → `-wip`）。
`git:<ref>` は `h4_wip/results/<名前>.json` をそのブランチから読む（取り込まない＝マージ不要）。
無い欄は `-` と表示する（空欄＝まだ測っていない）。

読み方の決め（事前に固定）: 手の評価は**帯別（close／mid／decided）の ΔG／ΔS の AUC**を主に読む。
帳簿は **rel_K の AUC**（全行／最後の自席ターンを外した行／最後の自席ターンだけ）で読み、
60 局の線形の相関は読まない（3 局に支配される・H-4g 診断）。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = "h4_wip/results"
DEFAULT_SRC = [os.path.join(ROOT, RES), "git:origin/claude/h4-real", "git:origin/claude/h4-syn",
               "git:origin/claude/h4-misc", "git:origin/claude/h4-rule-don-wip"]
H4F_TARGET = {"hit": 0.725, "bias": -0.01, "sigma_T": 1.42, "within1": 0.672, "mae": 0.91}   # H-4f の実（flat・mu）
N3_REAL = {"hit": 0.6721, "bias": 1.943, "sigma_T": 5.927}                                   # N-3 既定（実）
SRC = []


def load(name):
    """名前（拡張子なし）から json を 1 つ読む。見つからなければ `None`。"""
    for s in SRC:
        if s.startswith("git:"):
            try:
                out = subprocess.run(["git", "show", "%s:%s/%s.json" % (s[4:], RES, name)], cwd=ROOT,
                                     capture_output=True, text=True, timeout=60)
            except Exception:
                continue
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(out.stdout)
        else:
            p = os.path.join(s, name + ".json")
            if os.path.exists(p) and os.path.getsize(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
    return None


def _f(x, spec="%.4f"):
    return "-" if x is None else spec % x


def cb_row(j):
    th = j["summary"]["by_slope"]["theory"]; st = j["stats"]
    return [th["sign_accuracy"], th["bias"], th["sigma_T"], th["within1"], th["mae"], st["tau_capped"],
            st.get("plan_cut"), st.get("plan_n")]


def table_cb(sides=("real", "syn"), modes=("cuttable_forced", "rule_don", "rule", "rule_don_purse")):
    print("== 交点の橋  [的中 バイアス σ_T within1 MAE | 打ち切り | 守る側の計算が地平を縮めた計画／解いた計画]")
    for s in sides:
        for m in modes:
            j = load("cb_%s_%s" % (s, m))
            if j is None:
                print("%-5s %-16s -" % (s, m)); continue
            r = cb_row(j)
            print("%-5s %-16s %.4f %+.3f %.3f %.4f %.3f | %4d | %s／%s" % (s, m, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    j = load("cb_flatreal_rule_don")
    print("== 全局の再現（一律の値段・`--cut-price flat --cut-take mu`・実）と H-4f の数字")
    if j is None:
        print("   -（まだ測っていない）")
    else:
        r = cb_row(j)
        print("   今回  hit %.4f bias %+.3f σ_T %.3f within1 %.4f MAE %.3f" % tuple(r[:5]))
        print("   H-4f  hit %.4f bias %+.3f σ_T %.3f within1 %.4f MAE %.3f" % tuple(H4F_TARGET[k] for k in ("hit", "bias", "sigma_T", "within1", "mae")))
    print("== 必要量に対する耐久（残りターン別・理論／必要）")
    for s in sides:
        for m in ("cuttable_forced", "rule_don"):
            j = load("cb_%s_%s" % (s, m))
            if j:
                b = j["summary"]["theta_check"]["by_turns_left"]
                print("%-5s %-16s " % (s, m) + " ".join("%s:%.3f" % (k, v["theta_over_need"]) for k, v in b.items()))


def table_wc():
    print("== 決着前の較正  [LL ECE AUC Brier n]")
    for s in ("real", "syn"):
        for m in ("cuttable_forced", "rule_don"):
            w = load("wc_%s_%s" % (s, m))
            if w is None:
                print("%-5s %-16s -" % (s, m)); continue
            r = w["rel"]
            print("%-5s %-16s %.4f %.4f %.4f %.4f %d" % (s, m, r["logloss"], r["ece"], r["auc"], r["brier"], r["n"]))


def table_rl():
    print("== 相対の帳簿 rel_K の AUC  [全行／最後の自席ターンを外す／最後の自席ターンだけ] と rel_flat の AUC（60 局）")
    for s in ("real", "syn"):
        for m in ("cuttable_forced", "rule_don", "rule_don_nomirror"):
            l = load("rl_%s_%s" % (s, m))
            if l is None:
                print("%-5s %-18s -" % (s, m)); continue
            print("%-5s %-18s %.4f %.4f %.4f | rel_flat %.4f abs_flat %.4f | 打ち切り割合 %.4f" % (
                s, m, l["arms"]["rel_K"]["auc"], l["before"]["rel_K"]["auc"], l["last_turn"]["rel_K"]["auc"],
                l["arms"]["rel_flat"]["auc"], l["arms"]["abs_flat"]["auc"], l["capped_share"]))


def table_tb():
    print("== 線形の橋（手の評価）ΔG／ΔS の AUC  [全体 | close mid decided]  ※鏡あり／なしの差は decided 帯に集中するかを見る")
    for s in ("real", "syn"):
        for m in ("cuttable_forced", "rule_don", "rule_don_nomirror"):
            j = load("tb_%s_%s" % (s, m))
            if j is None:
                print("%-5s %-18s -" % (s, m)); continue
            sm = j["summary"]; bb = j.get("per_band", {})
            for k in ("dG", "dS"):
                print("%-5s %-18s %s %.3f | %s" % (s, m, k, sm[k]["auc"], " ".join(
                    "%s %.3f" % (b, bb[b][k]["auc"]) if b in bb and bb[b].get(k) else "%s -" % b for b in ("close", "mid", "decided"))))


def table_tl():
    print("== 遷移の帳簿（実）  [説明される割合／残差の割合／原因別の残差密度／攻撃の行の残差÷価格]")
    for m in ("cuttable_forced", "rule_don"):
        j = load("tl_real_%s" % m)
        if j is None:
            print("%-16s -" % m); continue
        fam = j.get("by_family", {}).get("attack", {})
        print("%-16s priced %.4f resid %.4f | per gap %s | attack resid/priced %s | gaps %d" % (
            m, j["priced_share"], j["resid_share"], j["resid_abs_per_gap"], fam.get("resid_over_priced"), j["gaps"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", default=[], help="ディレクトリ、または git:<ref>（繰り返し可）")
    a = ap.parse_args(argv)
    SRC[:] = a.src or DEFAULT_SRC
    print("N-3 既定（実・参考）:", N3_REAL)
    table_cb(); table_wc(); table_rl(); table_tb(); table_tl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
