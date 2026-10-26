"""一手先の価値だけで候補を選ぶと教師とどれだけ合うか（WP policy-after-desk の補助の計器・1 回きり）。

holdout（seed % 7）の方策点で、各候補の「打った後の盤面」に r24 の価値の頭を当て（手番の席から見た勝率）、
その最大の候補を 1 位として、§7.52 と同じ定義の 1 位一致・本当の読み違いの率を測る。
打った後に手番が相手へ移る候補（攻撃の宣言・ターン終了）と自分のままの候補を分けても数える。
"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.n_rel import N_TOK
from opcg_sim.learned.train import dump_io as DIO, n_rel_train as NT, n_rel_torch as TT
from opcg_sim.learned.train import n_rel_after_desk as AD
from opcg_sim.learned.train.n_eff_feat import build_eff_tables
from opcg_sim.learned.vocab import load_db

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
dirs = NT._expand([sys.argv[1]])
db = load_db(); stats, ab, abm, pwr, isl, vocab = build_eff_tables()
V, P, C = DIO.load_dump(dirs, vocab, n_tok=N_TOK)
ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
A = AD.load_after(dirs, sys.argv[2], DIO.default_cache_dir())
net = NT.NRelNet.load(sys.argv[3], tables=(stats, ab, abm, pwr, isl)); net.ablate = {"rel"}
Tn, _ = AD._make_classes()
tn = Tn(net); tn.eval()
vp = np.where(P["seed"] % 7 == 0)[0][:int(sys.argv[4]) if len(sys.argv) > 4 else 20000]
idx = TT.ragged_index(ptr[vp], P["len"][vp].astype(np.int64))
va = np.zeros(len(idx), np.float64)
with torch.no_grad():
    for s in range(0, len(idx), 4096):
        j = idx[s:s + 4096]
        va[s:s + len(j)] = tn.value(torch.from_numpy(np.asarray(A["a_sc"][j], np.float32)),
                                    torch.from_numpy(np.asarray(A["a_ci"][j], np.int64)),
                                    torch.from_numpy(np.asarray(A["a_tok"][j], np.float32)), None, None).numpy()
pend = np.asarray(A["a_pend"][idx])
if os.environ.get("PERSP") == "pending":           # 相手の席から符号化した候補は符号を返す
    va = np.where(pend > 0, va, -va)
win = np.asarray(A["a_win"][idx], np.float64); va = np.where(win != 0, win, va)
lens = P["len"][vp].astype(np.int64); seg = np.repeat(np.arange(len(vp)), lens)
pi = C["pi"][idx]; q = C["q"][idx]; nv = C["n"][idx]; at = C["at"][idx]
a_v = AD._seg_argmax(va, seg, len(vp)); a_t = AD._seg_argmax(pi.astype(np.float64), seg, len(vp))
mis = {"all": 0, "atk_atk": 0, "order": 0, "play_play": 0, "other": 0, "unvisited": 0}
for j in np.where(a_v != a_t)[0]:
    i_n, i_t = a_v[j], a_t[j]
    if nv[i_n] <= 0: mis["unvisited"] += 1; continue
    if q[i_t] - q[i_n] < AD.Q_GAP: continue
    mis["all"] += 1
    tn_, tt_ = int(at[i_n]), int(at[i_t])
    if tn_ in AD._AT_ATK and tt_ in AD._AT_ATK: mis["atk_atk"] += 1
    elif {tn_, tt_} <= set(AD._AT_ATK) | {AD._AT_PLAY} and (tn_ == AD._AT_PLAY) != (tt_ == AD._AT_PLAY): mis["order"] += 1
    elif tn_ == AD._AT_PLAY and tt_ == AD._AT_PLAY: mis["play_play"] += 1
    else: mis["other"] += 1
# 価値の一手先と教師の Q の相関（同じ点の中で・訪問のある候補だけ）
cor = []
for k in range(len(vp)):
    m = (seg == k) & (nv > 0)
    if m.sum() >= 3 and np.std(va[m]) > 0 and np.std(q[m]) > 0:
        cor.append(np.corrcoef(va[m], q[m])[0, 1])
# 教師の最善が「手番が自分のまま」の候補か「相手へ移る」候補かで分けた 1 位一致
tp = pend[a_t]
out = {"points": int(len(vp)), "top1_value_1ply": float((a_v == a_t).mean()),
       "top1_when_best_keeps_turn": float((a_v == a_t)[tp > 0].mean()), "top1_when_best_passes": float((a_v == a_t)[tp == 0].mean()),
       "persp": os.environ.get("PERSP", "actor"),
       **{f"mis_{k}": v / len(vp) for k, v in mis.items()},
       "pend_self_frac": float(pend.mean()), "q_corr_within_point_mean": float(np.mean(cor)), "q_corr_points": len(cor)}
print(json.dumps(out))
