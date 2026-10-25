"""周ごとの行（*.jsonl）→ 報告の表（markdown）と RESULT.json の数字。1 回きり。"""
import json, glob, os, sys
D = os.path.dirname(os.path.abspath(__file__))
ARMS = ["base_scratch", "attn_scratch", "base_r24", "attn_r24", "attn_r24_xrand"]
out = {}
for a in ARMS:
    f = os.path.join(D, a + ".jsonl")
    if not os.path.exists(f):
        continue
    rows = []
    for l in open(f):
        r = json.loads(l); v = r["val"]; p = r["train_probe"]
        rows.append({"ep": r["ep"], "train_ce_run": r["train_ce_run"], "train_probe_ce": p["ce"],
                     "train_probe_top1": p["top1"], "val_ce": v["ce"], "val_top1": v["top1"],
                     "val_mis": v["mis_all"], "val_mis_atk": v["mis_atk_atk"], "val_mis_order": v["mis_order"],
                     "val_mis_play": v["mis_play_play"], "val_mis_other": v["mis_other"],
                     "val_unvisited": v["mis_unvisited"], "val_vmse": r["val_vmse"], "sec": r["sec"],
                     "wp1x_norm": r.get("wp1x_norm")})
    out[a] = rows
if "--md" in sys.argv:
    f3 = lambda x: "—" if x is None else f"{x:.4f}"
    pc = lambda x: f"{100*x:.2f}%"
    print("| 組 | 周 | 訓練（周の平均） | 訓練（固定 8,000 点） | 検証の方策誤差 | 検証の 1 位一致 | 読み違い | 攻撃どうし | 順番 | 登場どうし | その他 | 価値の誤差 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, rows in out.items():
        for r in rows:
            print(f"| {a} | {r['ep'] if r['ep']>=0 else '前'} | {f3(r['train_ce_run'])} | {f3(r['train_probe_ce'])} | {f3(r['val_ce'])} | "
                  f"{r['val_top1']:.3f} | {pc(r['val_mis'])} | {pc(r['val_mis_atk'])} | {pc(r['val_mis_order'])} | "
                  f"{pc(r['val_mis_play'])} | {pc(r['val_mis_other'])} | {r['val_vmse']:.4f} |")
else:
    print(json.dumps(out, ensure_ascii=False))
