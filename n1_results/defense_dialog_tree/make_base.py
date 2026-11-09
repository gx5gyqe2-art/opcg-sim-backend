"""入れない形の比べの母集団: 既存の守りの監査（reaudit-r29f4・defense_all）を、旗ごとに分かれた局・分かれた点から先だけに絞る（§7.85・1 回きり）。
python n1_results/defense_dialog_tree/make_base.py  →  base_<m>/points.jsonl・verify.jsonl、続けて defense_audit.py summarize --points … --out base_<m>"""
import gzip, json, os
D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, "..", "reaudit_r29f4", "defense_all")
div = {d["seed"]: d["first"] for d in map(json.loads, gzip.open(os.path.join(D, "diverge_4462000.jsonl.gz"), "rt"))}
for m in ("life1", "all"):
    os.makedirs(f"{D}/base_{m}", exist_ok=True)
    keep = lambda d: d["seed"] in div and div[d["seed"]][m] is not None
    with open(f"{D}/base_{m}/points.jsonl", "w") as f:
        for l in open(f"{R}/points_4462000_300.jsonl"):
            d = json.loads(l)
            if "game" in d:
                if keep(d["game"]):
                    f.write(l)
            elif keep(d) and d["step"] >= div[d["seed"]][m]:
                f.write(l)
    with open(f"{D}/base_{m}/verify.jsonl", "w") as f:
        for l in open(f"{R}/verify.jsonl"):
            d = json.loads(l)
            if keep(d) and d["step"] >= div[d["seed"]][m]:
                f.write(l)
