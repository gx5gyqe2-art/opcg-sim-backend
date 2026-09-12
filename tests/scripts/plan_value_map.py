"""方針ヘッドの地図——V(盤面, 方針) を holdout の記録で読み、「どこでどの方針が勝ちやすいか」と
「CPU がそれに従っているか」を出す（計画 §20.10・WP `rs-plan-aux`・読み取り専用）。

入力は方針ヘッド付きのネット（`n_rel_train --plan-weight`）と v13/v14 の波。行は
  - 自席ターンの最初の main 行（＝そのターンの判断が見た盤面）: V_face／V_board／V_mixed と
    実際に打った方針（`plan_labels` と同じ分類）・レースの状態（`race_state` と同じ margin）
  - 相手ターンの最初の自分の行（カウンター窓など）: V_take／V_guard と実際に受けたか守ったか
holdout（`--holdout-mod`・seed%N==0）だけを読む＝訓練に使った行を地図にしない。

出すもの:
  attack.agree      … argmax(V_face, V_board, V_mixed) == 打った方針 の割合（ラベルのある行）
  attack.gap        … V[argmax] − V[打った方針] の平均（0 なら従っている・大きいほど「別の方針の方が
                       勝ちやすいとネット自身が言っている」）
  attack.by_zone    … lethal_margin 帯 × threat_margin 帯ごとの平均 V_face − V_board と argmax の分布
  attack.by_opp_hand… 相手手札帯ごとの同上
  attack.winrate    … 打った方針が argmax と一致した行／しなかった行の勝率（z>0）
  attack.calib      … クラスごとの mean V[played] と mean z（較正）
  defense.*         … V_guard − V_take を自分のライフ・ターン帯で／agree／calib

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/plan_value_map.py --net ~/nrel_r8.npz \\
    --in ~/n32_wave/w*/n_records --holdout-mod 7 --out ~/pvm_w32.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from race_state import _state, _extra as _race_extra, _margin_band, _hand_band, _turn_band  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned import n_rel_feat as NR  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

ATTACK = ("face", "board", "mixed")
DEFENSE = ("take", "guard")


def _extra(dd, n):
    """race_state の状態列に加えて、forward に要る scalars／card_idx／tokens を丸ごと持つ。"""
    sc, tk = _race_extra(dd, n)
    return {"race_sc": sc, "race_tk": tk,
            "sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32),
            "life0": np.asarray(dd["scalars"])[:n, 0].astype(np.float32)}


def _pad(sc, tok):
    """v13 の行（scalars 123・tokens 22×20）を現行の形へ 0 埋め（`dump_io.target_form` と同じ）。"""
    if sc.shape[1] < NL.D_SC:
        sc = np.concatenate([sc, np.zeros((sc.shape[0], NL.D_SC - sc.shape[1]), np.float32)], 1)
    if tok.shape[2] < NR.S_DIM:
        tok = np.concatenate([tok, np.zeros(tok.shape[:2] + (NR.S_DIM - tok.shape[2],), np.float32)], 2)
    return sc, tok


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def collect(net, rt, dirs, holdout_mod=7, limit_games=0, bs=512):
    cards = PL.Cards()
    att, dfn = [], []
    games = 0
    pend = []                              # forward 待ちの (kind, rec, sc, ci, tok)

    def flush():
        if not pend:
            return
        sc = np.stack([p[2] for p in pend]); ci = np.stack([p[3] for p in pend]); tok = np.stack([p[4] for p in pend])
        sc, tok = _pad(sc, tok)
        _v, vq = net.value_with_plan(sc, ci, tok, *NT.relations_or_zeros(net, ci, tok, rt))
        for (kind, rec, _s, _c, _t), q in zip(pend, vq):
            rec["vq"] = [float(x) for x in q]
            (att if kind == "a" else dfn).append(rec)
        pend.clear()

    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        labels, _unk = PL.label_game(rows, pol, ex["life0"], L, ptr, idx, cards)
        seen_own, seen_opp = set(), set()
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        for n, i in enumerate(idx):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) != 0 or (w, t) in seen_own:
                    continue
                seen_own.add((w, t))
                st = _state(ex["race_sc"][i], ex["race_tk"][i])
                rec = {"turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]), **st}
                pend.append(("a", rec, ex["sc"][i], ex["ci"][i], ex["tok"][i]))
            else:
                if (w, t) in seen_opp:
                    continue
                seen_opp.add((w, t))
                rec = {"turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]),
                       "my_life": int(round(float(ex["race_sc"][i][0]))),
                       "my_hand": int(round(float(ex["race_sc"][i][6])))}
                pend.append(("d", rec, ex["sc"][i], ex["ci"][i], ex["tok"][i]))
            if len(pend) >= bs:
                flush()
    flush()
    return att, dfn, games


def _attack_block(recs):
    out = {"turns": len(recs)}
    if not recs:
        return out
    lab = [r for r in recs if 0 <= r["played"] <= 2]
    out["labelled"] = len(lab)
    am = [int(np.argmax(r["vq"][:3])) for r in lab]
    out["argmax_dist"] = {c: sum(1 for a in am if a == k) / max(len(am), 1) for k, c in enumerate(ATTACK)}
    out["played_dist"] = {c: sum(1 for r in lab if r["played"] == k) / max(len(lab), 1) for k, c in enumerate(ATTACK)}
    agree = [a == r["played"] for a, r in zip(am, lab)]
    out["agree"] = _mean(agree)
    out["gap"] = _mean([r["vq"][a] - r["vq"][r["played"]] for a, r in zip(am, lab)])
    out["gap_gt_0.1"] = _mean([(r["vq"][a] - r["vq"][r["played"]]) > 0.1 for a, r in zip(am, lab)])
    out["winrate_agree"] = _mean([r["z"] > 0 for a, r in zip(am, lab) if a == r["played"]])
    out["winrate_disagree"] = _mean([r["z"] > 0 for a, r in zip(am, lab) if a != r["played"]])
    out["calib"] = {c: {"n": sum(1 for r in lab if r["played"] == k),
                        "mean_v_played": _mean([r["vq"][k] for r in lab if r["played"] == k]),
                        "mean_z": _mean([r["z"] for r in lab if r["played"] == k])}
                    for k, c in enumerate(ATTACK)}

    def zone(rs):
        if not rs:
            return {"n": 0}
        am_ = [int(np.argmax(r["vq"][:3])) for r in rs]
        lab_ = [(a, r) for a, r in zip(am_, rs) if 0 <= r["played"] <= 2]
        return {"n": len(rs), "v_face_minus_board": _mean([r["vq"][0] - r["vq"][1] for r in rs]),
                "v_face": _mean([r["vq"][0] for r in rs]), "v_board": _mean([r["vq"][1] for r in rs]),
                "v_mixed": _mean([r["vq"][2] for r in rs]),
                "argmax_face": _mean([a == 0 for a in am_]),
                "played_face": _mean([r["played"] == 0 for _a, r in lab_]) if lab_ else None,
                "agree": _mean([a == r["played"] for a, r in lab_]) if lab_ else None,
                "gap": _mean([r["vq"][a] - r["vq"][r["played"]] for a, r in lab_]) if lab_ else None,
                "winrate": _mean([r["z"] > 0 for r in rs])}
    out["by_zone"] = {}
    for lb in ("<=-2", "-1", "0", "+1", ">=+2"):
        for tb in ("<=-2", "-1", "0", "+1", ">=+2"):
            rs = [r for r in recs if _margin_band(r["lethal_margin"]) == lb and _margin_band(r["threat_margin"]) == tb]
            if rs:
                out["by_zone"][f"L{lb}|T{tb}"] = zone(rs)
    out["by_opp_hand"] = {b: zone([r for r in recs if _hand_band(r["opp_hand"]) == b]) for b in ("h0-1", "h2-3", "h4+")}
    out["by_turn"] = {b: zone([r for r in recs if _turn_band(r["turn"]) == b]) for b in ("t1-4", "t5-8", "t9+")}
    both = [r for r in recs if r["lethal_margin"] >= 0 and r["threat_margin"] >= 0]
    out["both_lethal"] = zone(both)
    out["lethal_only"] = zone([r for r in recs if r["lethal_margin"] >= 0 and r["threat_margin"] < 0])
    out["threat_only"] = zone([r for r in recs if r["lethal_margin"] < 0 and r["threat_margin"] >= 0])
    return out


def _defense_block(recs):
    out = {"turns": len(recs)}
    if not recs:
        return out
    lab = [r for r in recs if r["played"] in (3, 4)]
    out["labelled"] = len(lab)
    am = [3 + int(np.argmax(r["vq"][3:5])) for r in lab]
    out["argmax_guard"] = _mean([a == 4 for a in am])
    out["played_guard"] = _mean([r["played"] == 4 for r in lab])
    out["agree"] = _mean([a == r["played"] for a, r in zip(am, lab)])
    out["gap"] = _mean([r["vq"][a] - r["vq"][r["played"]] for a, r in zip(am, lab)])
    out["calib"] = {c: {"n": sum(1 for r in lab if r["played"] == 3 + k),
                        "mean_v_played": _mean([r["vq"][3 + k] for r in lab if r["played"] == 3 + k]),
                        "mean_z": _mean([r["z"] for r in lab if r["played"] == 3 + k])}
                    for k, c in enumerate(DEFENSE)}

    def zone(rs):
        if not rs:
            return {"n": 0}
        lab_ = [r for r in rs if r["played"] in (3, 4)]
        return {"n": len(rs), "v_guard_minus_take": _mean([r["vq"][4] - r["vq"][3] for r in rs]),
                "argmax_guard": _mean([r["vq"][4] > r["vq"][3] for r in rs]),
                "played_guard": _mean([r["played"] == 4 for r in lab_]) if lab_ else None,
                "winrate": _mean([r["z"] > 0 for r in rs])}
    out["by_life"] = {f"life{l}": zone([r for r in recs if r["my_life"] == l]) for l in (1, 2, 3, 4, 5)}
    out["by_turn"] = {b: zone([r for r in recs if _turn_band(r["turn"]) == b]) for b in ("t1-4", "t5-8", "t9+")}
    out["by_hand"] = {b: zone([r for r in recs if _hand_band(r["my_hand"]) == b]) for b in ("h0-1", "h2-3", "h4+")}
    return out


def _print(out):
    a = out["attack"]; d = out["defense"]
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"対局 {out['games']}・自席ターン {a['turns']}・相手ターン {d['turns']}")
    if a.get("labelled"):
        print(f"攻め: agree {f(a['agree'])}  gap {f(a['gap'])}  gap>0.1 {f(a['gap_gt_0.1'])}"
              f"  wr agree/disagree {f(a['winrate_agree'])}/{f(a['winrate_disagree'])}")
        print(f"  argmax {({k: round(v, 3) for k, v in a['argmax_dist'].items()})}  played {({k: round(v, 3) for k, v in a['played_dist'].items()})}")
        print("  calib " + "  ".join(f"{c}: n{v['n']} v {f(v['mean_v_played'])} z {f(v['mean_z'])}" for c, v in a["calib"].items()))
        for k in ("lethal_only", "threat_only", "both_lethal"):
            z = a[k]
            if z.get("n"):
                print(f"  {k:12} n {z['n']:6d}  V_face−V_board {f(z['v_face_minus_board'])}  argmax_face {f(z['argmax_face'])}"
                      f"  played_face {f(z['played_face'])}  agree {f(z['agree'])}  gap {f(z['gap'])}  wr {f(z['winrate'])}")
        for k, z in a["by_opp_hand"].items():
            if z.get("n"):
                print(f"  opp_hand {k:5} n {z['n']:6d}  V_face−V_board {f(z['v_face_minus_board'])}  argmax_face {f(z['argmax_face'])}  played_face {f(z['played_face'])}")
    if d.get("labelled"):
        print(f"守り: agree {f(d['agree'])}  gap {f(d['gap'])}  argmax_guard {f(d['argmax_guard'])}  played_guard {f(d['played_guard'])}")
        for k, z in d["by_life"].items():
            if z.get("n"):
                print(f"  {k:6} n {z['n']:6d}  V_guard−V_take {f(z['v_guard_minus_take'])}  argmax_guard {f(z['argmax_guard'])}  played_guard {f(z['played_guard'])}  wr {f(z['winrate'])}")
    print(f"  {out['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="方針ヘッド付きの NRel npz")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed%%N==0 の対局だけ（0＝全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    net = NL.NRelNet.load(args.net, (stats, ab, abm, pwr, isl))
    if not net.plan:
        raise SystemExit("このネットに方針ヘッド（plan_*）が無い")
    from opcg_sim.learned.vocab import load_db
    rt = NR.RelTable(NR.profile_table(load_db(), vocab))
    att, dfn, games = collect(net, rt, args.src, args.holdout_mod, args.limit_games)
    out = {"net": os.path.abspath(args.net), "src": [os.path.abspath(s) for s in args.src],
           "holdout_mod": args.holdout_mod, "games": games,
           "attack": _attack_block(att), "defense": _defense_block(dfn),
           "seconds": round(time.time() - t0, 1)}
    _print(out)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
