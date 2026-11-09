"""WP `value-skip`（`docs/search_efficiency_plan.md` §7.78・台帳 2+514）の数え。読み取り専用・1 回きりの計測＝テストは足さない
（ネットの形の切替そのもののテストは `tests/test_value_skip.py`）。

- `rows`: 波 82j のドンを動かした行での群の差（`don_jitter_rows.py rows` と同じ選び方・同じ式）を `n1_results/value_skip/` に書く。
  あわせて切替ありのネットでは、価値の出口の手前（tanh の前）の群の差を「共有表現からの道」と「入力の束からのまっすぐな道」に分ける。
- `weights`: 新しい重み（入力の束 415 列 → 出口）の大きさと、ドンの列に乗った重み。列ごとの「重み × その列の行ごとの散らばり
  （手番 3〜6 の行の標準偏差）」で並べ、ドンの列が何位にいるかを数える。
- `summary`: 試験台の出力（ドンの比・群の差・取り置きの誤差・訓練のログ・重み）を `summary.json` に束ねる。

    python tests/scripts/value_skip.py rows --nets opcg_sim/data/learned/nrel_r28x4.npz ~/ctrl/nrel_r29f4.ep0.npz ...
    python tests/scripts/value_skip.py weights --nets ~/vs/nrel_vskip.ep0.npz ~/vs/nrel_vskip.ep1.npz
    python tests/scripts/value_skip.py summary
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _bootstrap  # noqa: E402,F401
import don_trace as DT  # noqa: E402
import value_teach as VT  # noqa: E402

D = os.path.join("n1_results", "value_skip")
WAVE = "~/n82j/n_records/n82j_w*_p*/n_record_*.npz"
#: ドンが動かす列（§7.76 段 1 で 1 枚動かすと動いた列・スカラー）＋ドンの数そのもの。
DON_SC = ("don_active_me", "don_rested_me", "don_deck_me", "don_next_turn", "leader_power_max",
          "max_play_next_turn", "playable_chars_me", "don_addable", "leader_don_me")


def jitter_rows(wave):
    """82j のドンを動かした行（`don_jitter_rows.cmd_rows` と同じ選び方）。"""
    sc, ci, tok, sign, early, z, hold = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.expanduser(wave))):
        d = np.load(f, allow_pickle=True)
        dj, seed, turn, who = d["don_jitter"], d["seed"], d["turn"], d["who"]
        for s in np.unique(seed):
            idx = np.flatnonzero(seed == s)
            t, seat, sg = dj[idx[0]]
            if sg == 0:
                continue
            hit = idx[(turn[idx] == t) & (who[idx] == seat)]
            if not len(hit):
                continue
            i = int(hit[np.argmin(d["step"][hit])])
            sc.append(d["scalars"][i]); ci.append(d["card_idx"][i]); tok.append(d["tokens"][i])
            sign.append(int(sg)); early.append(int(t) <= 6); z.append(float(d["z"][i])); hold.append(int(s) % 7 == 0)
    return (np.array(sc), np.array(ci)[:, :22], np.array(tok), np.array(sign), np.array(early), np.array(z), np.array(hold))


def _diff(v, sign, m):
    a, b = v[m & (sign == 1)], v[m & (sign == -1)]
    se = math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
    return {"plus": round(float(a.mean()), 4), "minus": round(float(b.mean()), 4), "diff": round(float(a.mean() - b.mean()), 4),
            "ci95": round(1.96 * se, 4), "n": [int(len(a)), int(len(b))]}


def _load(path):
    from opcg_sim.learned import n_rel as NL
    VT.net_values(path, np.zeros((1, 127), np.float32), np.zeros((1, 22), np.int64), np.zeros((1, 22, 22), np.float32))
    return NL.NRelNet.load(path, tables=VT._G["tables"])


def _parts(net, sc, ci, tok, bs=256):
    """(共有表現からの道, まっすぐな道, 切片) の tanh の前の値を行ごとに。"""
    from opcg_sim.learned.train import n_rel_train as NT
    sh, sk = [], []
    tab = net.card_table()
    for s in range(0, len(sc), bs):
        a_, c_, t_ = sc[s:s + bs].astype(np.float32), ci[s:s + bs].astype(np.int64), tok[s:s + bs].astype(np.float32)
        h, present = net.tokens_forward(c_, t_, *NT.relations_or_zeros(net, c_, t_, VT._G["rt"]), tab)
        k = {}
        e = net.body(a_, h, present, k)
        sh.append((e @ net.Wv)[:, 0]); sk.append((k["z"] @ net.Wvz)[:, 0])
    return np.concatenate(sh), np.concatenate(sk), k["z"].shape[1]


def cmd_rows(a):
    sc, ci, tok, sign, early, z, hold = jitter_rows(a.wave)
    groups = {"turn3_6": early, "turn7_10": ~early, "all": np.ones(len(sign), bool),
              "turn3_6_holdout": early & hold, "turn3_6_trained": early & ~hold}
    res = {"rows": int(len(sign)), "outcome_z": {g: _diff(z, sign, m) for g, m in groups.items()}, "nets": {}, "logit_parts": {}}
    for n in a.nets:
        v = VT.net_values(n, sc, ci, tok)
        nm = os.path.basename(n)
        res["nets"][nm] = {g: _diff(v, sign, m) for g, m in groups.items()}
        net = _load(n)
        if getattr(net, "value_skip", False):
            sh, sk, _ = _parts(net, sc, ci, tok)
            res["logit_parts"][nm] = {"shared": _diff(sh, sign, early), "skip": _diff(sk, sign, early),
                                      "total": _diff(sh + sk, sign, early)}
        print(nm, {g: res["nets"][nm][g]["diff"] for g in groups}, res["logit_parts"].get(nm, {}), flush=True)
    print("z", {g: res["outcome_z"][g]["diff"] for g in groups})
    os.makedirs(D, exist_ok=True)
    json.dump(res, open(os.path.join(D, a.out_name), "w"), ensure_ascii=False, indent=1)


def z_names():
    """入力の束 415 列の名前（スカラー 127＋トークンの平均 144＋最大 144）。"""
    sc = DT.scalar_names()
    hn = ([f"t{j}" for j in range(48)] + [f"r_max{j}" for j in range(32)] + [f"r_mean{j}" for j in range(32)]
          + [f"c_max{j}" for j in range(32)])
    return sc + [f"tok_mean.{x}" for x in hn] + [f"tok_max.{x}" for x in hn]


def cmd_weights(a):
    sc, ci, tok, sign, early, _z, _hold = jitter_rows(a.wave)
    names = z_names()
    res = {"nets": {}}
    for n in a.nets:
        net = _load(n)
        nm = os.path.basename(n)
        if not getattr(net, "value_skip", False):
            continue
        W = net.Wvz[:, 0].astype(np.float64)
        # 行ごとの散らばり（手番 3〜6 の全てのドンを動かした行で、入力の束の列の標準偏差）
        from opcg_sim.learned.train import n_rel_train as NT
        m = early
        a_, c_, t_ = sc[m].astype(np.float32), ci[m].astype(np.int64), tok[m].astype(np.float32)
        Z = []
        tab = net.card_table()
        for s in range(0, len(a_), 256):
            h, present = net.tokens_forward(c_[s:s + 256], t_[s:s + 256], *NT.relations_or_zeros(net, c_[s:s + 256], t_[s:s + 256], VT._G["rt"]), tab)
            k = {}
            net.body(a_[s:s + 256], h, present, k)
            Z.append(k["z"])
        Z = np.concatenate(Z).astype(np.float64)
        sd = Z.std(0)
        imp = np.abs(W) * sd
        order = np.argsort(-imp)
        rank = {int(j): int(r) + 1 for r, j in enumerate(order)}
        sg = sign[m]
        dz = Z[sg == 1].mean(0) - Z[sg == -1].mean(0)                 # 入力の束の群の差（ドン 2 枚ぶん）
        contrib = W * dz
        don_idx = [j for j, x in enumerate(names) if x in DON_SC]
        res["nets"][nm] = {
            "l2": round(float(np.linalg.norm(W)), 5), "abs_sum": round(float(np.abs(W).sum()), 5), "abs_max": round(float(np.abs(W).max()), 5),
            "shared_exit_l2": round(float(np.linalg.norm(net.Wv)), 5),
            "by_part_abs_sum": {"scalars": round(float(np.abs(W[:127]).sum()), 5), "tok_mean": round(float(np.abs(W[127:271]).sum()), 5),
                                "tok_max": round(float(np.abs(W[271:]).sum()), 5)},
            "top20_by_weight_x_spread": [{"col": names[j], "w": round(float(W[j]), 5), "sd": round(float(sd[j]), 4),
                                          "w_x_sd": round(float(W[j] * sd[j]), 5)} for j in order[:20]],
            "don_cols": {names[j]: {"w": round(float(W[j]), 5), "sd": round(float(sd[j]), 4), "rank_w_x_sd": rank[j],
                                    "group_dz": round(float(dz[j]), 4), "contrib": round(float(contrib[j]), 5)} for j in don_idx},
            "skip_logit_group_diff": {"all_cols": round(float(contrib.sum()), 5), "don_cols": round(float(contrib[don_idx].sum()), 5),
                                      "scalars": round(float(contrib[:127].sum()), 5), "tokens": round(float(contrib[127:].sum()), 5)},
        }
        print(nm, json.dumps({k: res["nets"][nm][k] for k in ("l2", "abs_max", "by_part_abs_sum", "skip_logit_group_diff")}, ensure_ascii=False), flush=True)
        print("  don", {k: (v["w"], v["rank_w_x_sd"]) for k, v in res["nets"][nm]["don_cols"].items()}, flush=True)
    os.makedirs(D, exist_ok=True)
    json.dump(res, open(os.path.join(D, "weights.json"), "w"), ensure_ascii=False, indent=1)


def cmd_summary(a):
    out = {"tvval": {}, "train": {}, "rows_turn3_6": {}, "band": {}, "logit_parts": {}}
    for f in sorted(glob.glob(f"{D}/tvval_*.json")):
        for k, v in json.load(open(f))["nets"].items():
            out["tvval"][k] = {t: v[t]["mean"] for t in v} | {t + "_ratio": v[t]["ratio"] for t in v}
    for f in sorted(glob.glob(f"{D}/train_*.log")):
        name = os.path.basename(f)[6:-4]
        for line in open(f):
            if line.startswith("N_REL_TRAIN_EPOCH"):
                e = json.loads(line.split(" ", 1)[1])
                out["train"][f"nrel_{name}.ep{e['ep']}.npz"] = {k: round(e[k], 4) for k in ("train_mse", "val_vmse", "val_vsign", "val_pi_top1", "val_p_loss")}
            mm = re.match(r"value (\d+)行（val (\d+)）", line)
            if mm:
                out["train"][f"{name}_rows"] = {"value": int(mm[1]), "val": int(mm[2])}
    for f in sorted(glob.glob(f"{D}/jitter_rows*.json")):
        d = json.load(open(f))
        out["rows_turn3_6"]["outcome_z"] = d["outcome_z"]["turn3_6"]["diff"]
        for k, v in d["nets"].items():
            out["rows_turn3_6"][k] = {g: v[g]["diff"] for g in v}
        out["logit_parts"].update({k: {p: v[p]["diff"] for p in v} for k, v in d.get("logit_parts", {}).items()})
    for f in sorted(glob.glob(f"{D}/band_*.json")):
        out["band"][os.path.basename(f)[:-5]] = {k: {"v_mse": round(v["all"]["v_mse"], 4), "v_sign": round(v["all"]["v_sign"], 4)}
                                                for k, v in json.load(open(f))["nets"].items()}
    if os.path.exists(f"{D}/weights.json"):
        w = json.load(open(f"{D}/weights.json"))["nets"]
        out["weights"] = {k: {x: v[x] for x in ("l2", "abs_max", "by_part_abs_sum", "skip_logit_group_diff")} for k, v in w.items()}
    json.dump(out, open(f"{D}/summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["rows", "weights", "summary"])
    ap.add_argument("--wave", default=WAVE)
    ap.add_argument("--nets", nargs="*", default=[])
    ap.add_argument("--out-name", default="jitter_rows.json")
    a = ap.parse_args(argv)
    {"rows": cmd_rows, "weights": cmd_weights, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
