"""方針の地図の測り直し（WP `plan-bias` 段 1・`docs/search_efficiency_plan.md` §7.64・読み取り専用）。

2026-09 の 2 本の計器（`plan_value_map.py`＝方針ヘッドの地図・`time_plan_map.py`＝時間 × 方針と
「しない／できない」）は 2026-10-01 に退役した（履歴 a623c54e に在る）。段 1 で要る部分だけを
1 本にまとめ直した＝**数え方は旧い 2 本と同じ**（行の取り方・方針の分類・帯・選べたかの読み）。
1 回きりの測定の道具なのでテストは足さない（CLAUDE.md の理論の道具の規約）。

行（holdout＝seed%N==0 の対局だけ）:
  攻め … 自席ターンの最初の main 行。V_face／V_board／V_mixed と、そのターンに打った方針
         （`plan_labels.label_game`）・レースの状態（旧 `race_state._state`）・候補一覧から
         「リーダーを殴る手／キャラを殴るか除去する手が在ったか」
  守り … 相手ターンの最初の自分の行。V_take／V_guard と、受けたか守ったか・符号化から
         「手札のカウンター値の合計／起動中のブロッカー」と飛んでくる攻撃の超過パワー
残りターン数は記録から後付けした真値（`time_labels.label_game` の列 0＝決着までの自席ターン数）。

出すもの:
  r8 の表 … 攻めの agree／gap／gap>0.1／勝率（一致・不一致）／argmax と打った方針の分布／
             レースの状態別（リーサル圏だけ・被リーサル圏だけ・両方の圏）／ターン帯／較正、
             守りの agree／gap／argmax_guard／played_guard／自分のライフ別
  時計の表 … 決着まで自席 2 ターン以内（t<=2）・3〜4・5+ の帯ごとに、上と同じ量と
             「その方針を打った局の勝率」。攻めは全行と「両方選べた行」、守りは全行と
             「守る手段が在った行」「止めるだけの量が在った行」

  OPCG_LOG_SILENT=1 python tests/scripts/plan_bias_map.py --net ~/nrel_pmap.npz \\
    --in ~/n58/n_records/n58_w*_p[0-3] ~/n59/n_records/n59_w*_p[0-3] --holdout-mod 7 --out pbm.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned import n_rel_feat as NR  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

ATTACK = ("face", "board", "mixed")
DEFENSE = ("take", "guard")
CLOCK_BANDS = ("t<=2", "t3-4", "t5+")
# tokens の列（`n_rel_feat.S_COLS`）・枠（0 自L・1 相L・2〜6 自場・7〜11 相場・12〜21 手札）
S_PWR, S_CAN, S_BLK, S_CNT, S_CHAR = 0, 5, 6, 7, 18
SLOT_OWN, SLOT_OPP, SLOT_HAND = slice(2, 7), slice(7, 12), slice(12, 22)
PWR_SCALE = 10000.0
COUNTER_SCALE = 2000.0          # `counter_value` は `min(counter/2000, 2.5)`


def margin_band(m):
    return "<=-2" if m <= -2 else "-1" if m == -1 else "0" if m == 0 else "+1" if m == 1 else ">=+2"


def turn_band(t):
    return "t1-4" if t <= 4 else "t5-8" if t <= 8 else "t9+"


def clock_band(t):
    return "t<=2" if t <= 2.5 else ("t3-4" if t <= 4.5 else "t5+")


def race(sc, tk):
    """自席ターン開始の main 行 → レースの状態（旧 `race_state._state` と同じ式）。"""
    own, opp = tk[SLOT_OWN], tk[SLOT_OPP]
    own_ch, opp_ch = own[:, S_CHAR] > 0.5, opp[:, S_CHAR] > 0.5
    my_atk = int(tk[0, S_CAN] > 0.5) + int((own_ch & (own[:, S_CAN] > 0.5)).sum())
    opp_atk = 1 + int(opp_ch.sum())
    my_blk = int((own_ch & (own[:, S_BLK] > 0.5)).sum())
    opp_blk = int((opp_ch & (opp[:, S_BLK] > 0.5)).sum())
    my_life, opp_life = int(round(sc[0])), int(round(sc[1]))
    return {"my_life": my_life, "opp_life": opp_life,
            "my_hand": int(round(sc[6])), "opp_hand": int(round(sc[7])),
            "lethal_margin": my_atk - opp_blk - (opp_life + 1),
            "threat_margin": opp_atk - my_blk - (my_life + 1)}


def incoming(tk):
    """相手ターン中の自分の行 → 飛んでくる攻撃の超過パワー（旧 `race_state._incoming`）。"""
    opp = tk[SLOT_OPP]
    cand = [float(tk[1, S_PWR])]
    ch = opp[:, S_CHAR] > 0.5
    if ch.any():
        cand.append(float(opp[ch, S_PWR].max()))
    top = max(cand) * PWR_SCALE
    if top <= 0.0:
        return None
    return top - float(tk[0, S_PWR]) * PWR_SCALE


def guard_means(tk):
    """手札のカウンター値の合計（power 単位）と、起動中のブロッカーが居るか（旧 `time_plan_map`）。"""
    csum = float(tk[SLOT_HAND][:, S_CNT].sum()) * COUNTER_SCALE
    return csum, bool((tk[SLOT_OWN][:, S_BLK] > 0.5).any())


def attack_avail(pol, L, ptr, i, u2c, cards):
    k, base = int(L[i]), int(ptr[i])
    face = board = False
    for j in range(base, base + k):
        cl = PL.move_class(json.loads(pol["pol_sig"][j]), u2c, cards)
        face |= cl == "face"
        board |= cl == "board"
    return face, board


def _pad(sc, tok):
    if sc.shape[1] < NL.D_SC:
        sc = np.concatenate([sc, np.zeros((sc.shape[0], NL.D_SC - sc.shape[1]), np.float32)], 1)
    if tok.shape[2] < NR.S_DIM:
        tok = np.concatenate([tok, np.zeros(tok.shape[:2] + (NR.S_DIM - tok.shape[2],), np.float32)], 2)
    return sc, tok


def _extra(dd, n):
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    return {"sc": sc, "tok": tok, "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "life0": sc[:, 0].copy(), "lives": sc[:, 0:2].copy()}


def collect(net, rt, dirs, holdout_mod=7, limit_games=0, bs=512):
    cards = PL.Cards()
    att, dfn, pend = [], [], []
    games = 0

    def flush():
        if not pend:
            return
        sc = np.stack([p[2] for p in pend]); ci = np.stack([p[3] for p in pend]); tok = np.stack([p[4] for p in pend])
        v, vq = net.value_with_plan(sc, ci, tok, *NT.relations_or_zeros(net, ci, tok, rt))
        for (kind, rec, *_r), q, v0 in zip(pend, vq, np.asarray(v).reshape(-1)):
            rec["vq"] = [float(x) for x in q]
            rec["v"] = float(v0)
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
        tm, tmask = TL.label_game(rows, ex["lives"], idx)
        u2c = PL.uuid_map(pol, L, ptr, idx)
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen = set()
        for n, i in enumerate(idx):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1:
                continue
            own = PL.is_own_turn(w, t)
            if own and int(rows["kind"][i]) != 0:
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            tk = ex["tok"][i]
            rec = {"seed": seed, "step": int(rows["step"][i]), "who": w, "turn": t,
                   "z": zs.get(w, 0.0), "played": int(labels[n]),
                   "clock": (float(tm[n][0] * TL.T_SCALE) if tmask[n] else None)}
            if own:
                rec.update(race(ex["sc"][i], tk))
                rec["face_avail"], rec["board_avail"] = attack_avail(pol, L, ptr, i, u2c, cards)
            else:
                csum, blk = guard_means(tk)
                over = incoming(tk)
                rec.update(my_life=int(round(float(ex["sc"][i][0]))), my_hand=int(round(float(ex["sc"][i][6]))),
                           counter_sum=csum, blocker=blk, atk_over=over,
                           guard_avail=bool(csum > 0.0 or blk),
                           guard_enough=bool(blk or (over is not None and (over <= 0.0 or csum >= over))))
            pend.append(("a" if own else "d", rec, ex["sc"][i], ex["ci"][i], tk))
            if len(pend) >= bs:
                flush()
    flush()
    return att, dfn, games


def _m(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def _wr(rs):
    return _m(r["z"] > 0 for r in rs)


def att_zone(rs):
    """攻めの帯 1 つ（r8 の表の列＋打った方針ごとの勝率）。"""
    if not rs:
        return {"n": 0}
    am = [int(np.argmax(r["vq"][:3])) for r in rs]
    lab = [(a, r) for a, r in zip(am, rs) if 0 <= r["played"] <= 2]
    out = {"n": len(rs), "labelled": len(lab),
           "v_face_minus_board": _m(r["vq"][0] - r["vq"][1] for r in rs),
           "v_face_minus_best_other": _m(r["vq"][0] - max(r["vq"][1], r["vq"][2]) for r in rs),
           "argmax": {c: _m(a == k for a in am) for k, c in enumerate(ATTACK)},
           "played": {c: _m(r["played"] == k for _a, r in lab) for k, c in enumerate(ATTACK)},
           "agree": _m(a == r["played"] for a, r in lab),
           "gap": _m(r["vq"][a] - r["vq"][r["played"]] for a, r in lab),
           "gap_gt_0.1": _m((r["vq"][a] - r["vq"][r["played"]]) > 0.1 for a, r in lab),
           "winrate": _wr(rs),
           "winrate_agree": _wr([r for a, r in lab if a == r["played"]]),
           "winrate_disagree": _wr([r for a, r in lab if a != r["played"]]),
           "winrate_by_played": {c: _wr([r for _a, r in lab if r["played"] == k]) for k, c in enumerate(ATTACK)},
           "n_by_played": {c: sum(1 for _a, r in lab if r["played"] == k) for k, c in enumerate(ATTACK)},
           "calib": {c: {"n": sum(1 for _a, r in lab if r["played"] == k),
                         "v": _m(r["vq"][k] for _a, r in lab if r["played"] == k),
                         "z": _m(r["z"] for _a, r in lab if r["played"] == k)} for k, c in enumerate(ATTACK)}}
    # 「face を勧めているのに face を打っていない」行（段 2 の手当ての的）
    fr = [(a, r) for a, r in lab if a == 0]
    out["face_rec_n"] = len(fr)
    out["face_rec_played_face"] = _m(r["played"] == 0 for _a, r in fr)
    return out


def def_zone(rs):
    if not rs:
        return {"n": 0}
    lab = [r for r in rs if r["played"] in (3, 4)]
    am = [3 + int(r["vq"][4] > r["vq"][3]) for r in lab]
    gr = [(a, r) for a, r in zip(am, lab) if a == 4]
    return {"n": len(rs), "labelled": len(lab),
            "v_guard_minus_take": _m(r["vq"][4] - r["vq"][3] for r in rs),
            "argmax_guard": _m(r["vq"][4] > r["vq"][3] for r in rs),
            "played_guard": _m(r["played"] == 4 for r in lab),
            "agree": _m(a == r["played"] for a, r in zip(am, lab)),
            "gap": _m(r["vq"][a] - r["vq"][r["played"]] for a, r in zip(am, lab)),
            "gap_gt_0.1": _m((r["vq"][a] - r["vq"][r["played"]]) > 0.1 for a, r in zip(am, lab)),
            "winrate": _wr(rs),
            "winrate_by_played": {"take": _wr([r for r in lab if r["played"] == 3]),
                                  "guard": _wr([r for r in lab if r["played"] == 4])},
            "n_by_played": {"take": sum(1 for r in lab if r["played"] == 3),
                            "guard": sum(1 for r in lab if r["played"] == 4)},
            "calib": {c: {"n": sum(1 for r in lab if r["played"] == 3 + k),
                          "v": _m(r["vq"][3 + k] for r in lab if r["played"] == 3 + k),
                          "z": _m(r["z"] for r in lab if r["played"] == 3 + k)} for k, c in enumerate(DEFENSE)},
            "guard_rec_n": len(gr), "guard_rec_played_guard": _m(r["played"] == 4 for _a, r in gr)}


def report(att, dfn):
    both = lambda r: r["lethal_margin"] >= 0 and r["threat_margin"] >= 0  # noqa: E731
    out = {"attack": att_zone(att), "defense": def_zone(dfn)}
    out["attack_zones"] = {
        "lethal_only": att_zone([r for r in att if r["lethal_margin"] >= 0 and r["threat_margin"] < 0]),
        "threat_only": att_zone([r for r in att if r["lethal_margin"] < 0 and r["threat_margin"] >= 0]),
        "both_lethal": att_zone([r for r in att if both(r)])}
    out["attack_by_turn"] = {b: att_zone([r for r in att if turn_band(r["turn"]) == b]) for b in ("t1-4", "t5-8", "t9+")}
    out["defense_by_life"] = {f"life{l}": def_zone([r for r in dfn if r["my_life"] == l]) for l in range(1, 6)}
    out["defense_by_turn"] = {b: def_zone([r for r in dfn if turn_band(r["turn"]) == b]) for b in ("t1-4", "t5-8", "t9+")}
    ck = lambda rs, b: [r for r in rs if r["clock"] is not None and clock_band(r["clock"]) == b]  # noqa: E731
    attb = [r for r in att if r["face_avail"] and r["board_avail"]]
    out["clock"] = {
        "attack": {b: att_zone(ck(att, b)) for b in CLOCK_BANDS},
        "attack_both_avail": {b: att_zone(ck(attb, b)) for b in CLOCK_BANDS},
        "attack_both_lethal": {b: att_zone(ck([r for r in att if both(r)], b)) for b in CLOCK_BANDS},
        "defense": {b: def_zone(ck(dfn, b)) for b in CLOCK_BANDS},
        "defense_can": {b: def_zone(ck([r for r in dfn if r["guard_avail"]], b)) for b in CLOCK_BANDS},
        "defense_enough": {b: def_zone(ck([r for r in dfn if r["guard_enough"]], b)) for b in CLOCK_BANDS},
        "defense_enough_by_life": {f"life{l}": def_zone(ck([r for r in dfn if r["guard_enough"] and r["my_life"] == l], "t<=2"))
                                   for l in range(1, 6)},
    }
    # 選べたか（board を打った行のうち face も在った割合・受けた行のうち守る手段／量が在った割合）
    cap = {}
    for b in CLOCK_BANDS:
        ab = [r for r in ck(att, b) if r["played"] == 1]
        tk = [r for r in ck(dfn, b) if r["played"] == 3]
        cap[b] = {"board_played": len(ab), "board_played_face_avail": _m(r["face_avail"] for r in ab),
                  "take_played": len(tk), "take_played_guard_avail": _m(r["guard_avail"] for r in tk),
                  "take_played_guard_enough": _m(r["guard_enough"] for r in tk)}
    out["capability"] = cap
    return out


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def _print(out):
    a, d = out["attack"], out["defense"]
    print(f"対局 {out['games']}・自席ターン {a['n']}・相手ターン {d['n']}")
    print(f"攻め: agree {f3(a['agree'])} gap {f3(a['gap'])} gap>0.1 {f3(a['gap_gt_0.1'])} "
          f"wr 一致/不一致 {f3(a['winrate_agree'])}/{f3(a['winrate_disagree'])}")
    print(f"  argmax {({k: round(v, 3) for k, v in a['argmax'].items()})} played {({k: round(v, 3) for k, v in a['played'].items()})}")
    print("  calib " + "  ".join(f"{c}: n{v['n']} v {f3(v['v'])} z {f3(v['z'])}" for c, v in a["calib"].items()))
    for k, z in list(out["attack_zones"].items()) + list(out["attack_by_turn"].items()):
        if z.get("n"):
            print(f"  {k:12} n {z['n']:6d} Vf−Vb {f3(z['v_face_minus_board'])} argmax_face {f3(z['argmax']['face'])} "
                  f"played_face {f3(z['played']['face'])} agree {f3(z['agree'])} gap {f3(z['gap'])} wr {f3(z['winrate'])}")
    print(f"守り: agree {f3(d['agree'])} gap {f3(d['gap'])} argmax_guard {f3(d['argmax_guard'])} played_guard {f3(d['played_guard'])}")
    print("  calib " + "  ".join(f"{c}: n{v['n']} v {f3(v['v'])} z {f3(v['z'])}" for c, v in d["calib"].items()))
    for k, z in out["defense_by_life"].items():
        if z.get("n"):
            print(f"  {k:6} n {z['n']:6d} Vg−Vt {f3(z['v_guard_minus_take'])} argmax_guard {f3(z['argmax_guard'])} "
                  f"played_guard {f3(z['played_guard'])} wr {f3(z['winrate'])}")
    print("--- 時計（決着までの自席ターン数・真値）---")
    for name, blk in out["clock"].items():
        print(name)
        for b, z in blk.items():
            if not z.get("n"):
                continue
            if "argmax" in z:
                print(f"  {b:6} n {z['n']:6d} Vf−Vb {f3(z['v_face_minus_board'])} argmax {({k: round(v, 3) for k, v in z['argmax'].items()})} "
                      f"played {({k: round(v or 0, 3) for k, v in z['played'].items()})} agree {f3(z['agree'])} gap {f3(z['gap'])} "
                      f"wr打った face/board/mixed {f3(z['winrate_by_played']['face'])}/{f3(z['winrate_by_played']['board'])}/{f3(z['winrate_by_played']['mixed'])} "
                      f"face勧め→face {f3(z['face_rec_played_face'])}")
            else:
                print(f"  {b:6} n {z['n']:6d} Vg−Vt {f3(z['v_guard_minus_take'])} argmax_guard {f3(z['argmax_guard'])} "
                      f"played_guard {f3(z['played_guard'])} agree {f3(z['agree'])} gap {f3(z['gap'])} "
                      f"wr take/guard {f3(z['winrate_by_played']['take'])}/{f3(z['winrate_by_played']['guard'])} "
                      f"guard勧め→guard {f3(z['guard_rec_played_guard'])}")
    print(f"選べたか {json.dumps(out['capability'], ensure_ascii=False)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="方針ヘッド付きの NRel npz")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    tabs = build_eff_tables()
    net = NL.NRelNet.load(args.net, tabs[:5])
    if not net.plan:
        raise SystemExit("このネットに方針ヘッド（plan_*）が無い")
    rt = None
    if "rel" not in (net.ablate or ()):
        from opcg_sim.learned.vocab import load_db
        rt = NR.RelTable(NR.profile_table(load_db(), tabs[5]))
    att, dfn, games = collect(net, rt, args.src, args.holdout_mod, args.limit_games)
    out = {"net": os.path.basename(args.net), "src_n": len(args.src), "holdout_mod": args.holdout_mod,
           "games": games, **report(att, dfn), "seconds": round(time.time() - t0, 1)}
    _print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
