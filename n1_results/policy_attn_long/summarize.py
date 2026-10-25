"""周ごとの行（*.jsonl）→ 報告の表（markdown・--md）／差の表（--diff）／RESULT.json 用の数字。1 回きり。"""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
ARMS = ["base_scratch_w53", "attn_scratch_w53", "base_scratch_w5354", "attn_scratch_w5354", "base_r24_w5354"]


def load(a):
    f = os.path.join(D, a + ".jsonl")
    if not os.path.exists(f):
        return None
    rows = []
    for l in open(f):
        r = json.loads(l); v = r["val"]; p = r["train_probe"]
        rows.append({"ep": r["ep"], "train_ce_run": r["train_ce_run"], "train_probe_ce": p["ce"],
                     "val_ce": v["ce"], "val_top1": v["top1"], "val_mis": v["mis_all"],
                     "val_mis_atk": v["mis_atk_atk"], "val_mis_order": v["mis_order"],
                     "val_mis_play": v["mis_play_play"], "val_mis_other": v["mis_other"],
                     "val_vmse": r["val_vmse"], "sec": r["sec"]})
    return rows


out = {a: r for a in ARMS if (r := load(a)) is not None}
f4 = lambda x: "—" if x is None else f"{x:.4f}"
pc = lambda x: f"{100*x:.2f}%"
if "--md" in sys.argv:
    print("| 組 | 周 | 訓練（周の平均） | 訓練（固定 8,000 点） | 検証の方策誤差 | 1 位一致 | 読み違い | 攻撃どうし | 順番 | 登場どうし | その他 | 価値の誤差 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, rows in out.items():
        for r in rows:
            print(f"| {a} | {r['ep'] if r['ep']>=0 else '前'} | {f4(r['train_ce_run'])} | {f4(r['train_probe_ce'])} | {f4(r['val_ce'])} | "
                  f"{r['val_top1']:.3f} | {pc(r['val_mis'])} | {pc(r['val_mis_atk'])} | {pc(r['val_mis_order'])} | "
                  f"{pc(r['val_mis_play'])} | {pc(r['val_mis_other'])} | {r['val_vmse']:.4f} |")
elif "--diff" in sys.argv:
    for m in ("w53", "w5354"):
        b, t = out.get("base_scratch_" + m), out.get("attn_scratch_" + m)
        if not b or not t:
            continue
        print(f"## {m}")
        print("| 周 | 今の頭 | 注意の頭 | 差（検証） | 差（訓練・固定点） | 差（読み違い） | 差（価値） |")
        print("|---|---|---|---|---|---|---|")
        for rb, rt in zip(b[1:], t[1:]):
            print(f"| {rb['ep']} | {rb['val_ce']:.4f} | {rt['val_ce']:.4f} | {rt['val_ce']-rb['val_ce']:+.4f} | "
                  f"{rt['train_probe_ce']-rb['train_probe_ce']:+.4f} | {100*(rt['val_mis']-rb['val_mis']):+.2f} 点 | "
                  f"{rt['val_vmse']-rb['val_vmse']:+.4f} |")
else:
    print(json.dumps(out, ensure_ascii=False))
