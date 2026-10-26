"""policy-oracle-hand-desk の周ごとの数字（`*.jsonl`）→ 報告の表（markdown）。

  python n1_results/policy_oracle_hand_desk/summarize.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [("base_scratch", "今の入力・一から"), ("true_scratch", "本当の手札・一から"),
        ("base_r24", "今の入力・r24 から"), ("true_r24", "本当の手札・r24 から"),
        ("shuffle_r24", "対照（混ぜた列）・r24 から"), ("shuffle_scratch", "対照（混ぜた列）・一から")]


def rows(name):
    p = os.path.join(HERE, name + ".jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pct(x):
    return f"{100 * x:.2f}%"


def line(label, r, key):
    v = r[key]
    if v is None:
        return None
    vm = r["val_vmse" + ("" if key == "val" else key[3:])]
    ep = "前" if r["ep"] < 0 else str(r["ep"])
    return (f"| {label} | {ep} | {v['ce']:.4f} | {v['top1']:.3f} | {pct(v['mis_all'])} | {pct(v['mis_atk_atk'])} | "
            f"{pct(v['mis_order'])} | {pct(v['mis_play_play'])} | {pct(v['mis_other'])} | {vm:.3f} |")


def main():
    for key, title in (("val", "全体"), ("val_atk", "攻め（決める席が手番）"), ("val_def", "守り（決める席が手番でない）")):
        print(f"\n### {title}\n")
        print("| 組 | 周 | 方策誤差 | 1 位一致 | 読み違い | 攻撃どうし | 順番 | 登場どうし | その他 | 価値の誤差 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, label in ARMS:
            for i, r in enumerate(rows(name)):
                s = line(label if i == 0 else "", r, key)
                if s:
                    print(s)
    print("\n### 訓練側（固定 8,000 点）\n")
    for name, label in ARMS:
        rr = rows(name)
        if rr:
            print(f"- {label}: " + " / ".join(f"{r['train_probe']['ce']:.4f}" for r in rr))


if __name__ == "__main__":
    sys.exit(main())
