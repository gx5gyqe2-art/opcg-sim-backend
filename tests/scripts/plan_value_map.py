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

**リーダー／デッキ別の層別**（2026-09-12・ユーザ指摘「しきい値はリーダー＝デッキの中身で変わる。
平均値に強制するのは誤った学習につながる」）: 全リーダーを畳んだ平均は「集団平均」でしかないので、
**対局メタ（`meta_games.json` の両席のリーダー）と `deck_kinds`（除去の型・色・注入テンプレート）で
割り直す**。守りの「しきい値」は群ごとに `V_guard − V_take` のライフ別の形を出し、**山（peak_life）の
散らばり**を見る＝散れば平均は意味を持たない。出すもの:
  defense.by_leader_life／by_leader_colors／by_deck_template／by_deck_rate … 群ごとのライフ曲線
  defense.by_leader         … 出現数の多い上位リーダー個別のライフ曲線（`--leader-top`・`--min-n`）
  defense.peak_spread       … 群／リーダーごとの peak_life の分布（**しきい値が共通か個別かの答え**）
  attack.by_leader_*        … 同じ層別を両方の圏の V_face − V_board にかけたもの

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
LIVES = (1, 2, 3, 4, 5)
#: `deck_kinds` の注入テンプレート名（`deck_roles.TEMPLATE_NAMES` と同じ並び・層別の軸）
DECK_TEMPLATES = ("reduce_then_ko", "lock_and_attack", "deck_out", "trigger_removal",
                  "on_attack_removal", "activate_main_removal", "bounce", "trash", "event_removal")


def game_leaders(dirs):
    """`meta_games.json`（part ごとの sidecar）→ {seed: (p1 のリーダー, p2 のリーダー)}。

    seed 帯は波・シャードで重ならない（§6 の台帳で払い出す）ので 1 本の dict にまとめて良い。
    sidecar が無い part（打ち切りなど）はその対局が層別から落ちるだけ（`leader` が None）。
    """
    out = {}
    miss = 0
    for d in dirs:
        f = os.path.join(d, "meta_games.json")
        if not os.path.exists(f):
            miss += 1
            continue
        with open(f) as fh:
            meta = json.load(fh)
        for g in meta.get("games") or ():
            ld = g.get("leaders") or [None, None]
            out[int(g["seed"])] = (ld[0], ld[1])
    return out, miss


class Leaders:
    """リーダー ID → ライフ枚数・色・パワー（マスター単位でキャッシュ）。"""

    def __init__(self, db):
        self.db = db
        self._t = {}

    def info(self, cid):
        if cid not in self._t:
            m = self.db.get_card(cid) if cid else None
            if m is None:
                self._t[cid] = None
            else:
                cols = tuple(sorted(getattr(c, "name", str(c)) for c in (getattr(m, "colors", ()) or ())))
                self._t[cid] = {"life": int(getattr(m, "life", 0) or 0), "colors": cols,
                                "power": int(getattr(m, "power", 0) or 0),
                                "n_colors": len(cols)}
        return self._t[cid]


def deck_info(s):
    """`deck_kinds` の JSON → 層別の軸（色・注入テンプレート・注入率）。

    **型を持たない行（`deck_kinds` が無い波・`"{}"`）は空の dict を返す**＝デッキの軸から
    そのまま落ちる。0 値の辞書を返すと「注入率 0 のデッキ」として混ざるので返さない。
    """
    try:
        j = json.loads(s) if s else {}
    except Exception:
        return {}
    if not isinstance(j, dict) or not j:
        return {}
    return {"deck_colors": tuple(sorted(j.get("colors") or ())),
            "deck_templates": tuple(sorted(j.get("templates") or ())),
            "deck_rate": int(j.get("rate") or 0),
            "deck_n_inj": int(j.get("n") or 0)}


def _rate_band(r):
    return "rate0" if r <= 0 else "rate1-9" if r < 10 else "rate10-19" if r < 20 else "rate20+"


def life_curve(recs, min_n=1):
    """レコード群 → ライフ別の V_guard − V_take の曲線と山（peak_life）。

    `peak_life` は「n ≥ min_n のライフのうち V_guard − V_take が最大のライフ」。曲線が平ら
    （最大 − 最小 < flat_eps）かどうかも返す＝**しきい値が実在するか**の目安。
    """
    out = {"n": len(recs), "by_life": {}}
    if not recs:
        return out
    best = None
    vals = []
    for l in LIVES:
        sub = [r for r in recs if r["my_life"] == l]
        if not sub:
            continue
        d = float(np.mean([r["vq"][4] - r["vq"][3] for r in sub]))
        row = {"n": len(sub), "v_guard_minus_take": d,
               "argmax_guard": float(np.mean([r["vq"][4] > r["vq"][3] for r in sub])),
               "played_guard": (float(np.mean([r["played"] == 4 for r in sub if r["played"] in (3, 4)]))
                                if [r for r in sub if r["played"] in (3, 4)] else None),
               "winrate": float(np.mean([r["z"] > 0 for r in sub]))}
        out["by_life"][f"life{l}"] = row
        if len(sub) >= min_n:
            vals.append(d)
            if best is None or d > best[1]:
                best = (l, d)
    out["peak_life"] = (best[0] if best else None)
    out["peak_value"] = (best[1] if best else None)
    out["span"] = (float(max(vals) - min(vals)) if len(vals) >= 2 else None)
    return out


def both_lethal_block(recs):
    """両方の圏（お互いにリーサル圏）の攻めの向き（群ごとの層別に使う）。"""
    rs = [r for r in recs if r.get("lethal_margin", -9) >= 0 and r.get("threat_margin", -9) >= 0]
    if not rs:
        return {"n": 0}
    lab = [r for r in rs if 0 <= r["played"] <= 2]
    return {"n": len(rs),
            "v_face_minus_board": float(np.mean([r["vq"][0] - r["vq"][1] for r in rs])),
            "argmax_face": float(np.mean([int(np.argmax(r["vq"][:3])) == 0 for r in rs])),
            "played_face": (float(np.mean([r["played"] == 0 for r in lab])) if lab else None),
            "winrate": float(np.mean([r["z"] > 0 for r in rs]))}


def _group_block(recs, min_n):
    """1 群の出力（守りのライフ曲線＋攻めの両方の圏）。"""
    return {"defense": life_curve([r for r in recs if r["kind"] == "d"], min_n),
            "attack_both_lethal": both_lethal_block([r for r in recs if r["kind"] == "a"])}


def strat_blocks(recs, leader_top=16, min_n=60):
    """リーダー／デッキの軸で割った群の表と、peak_life の散らばり。"""
    out = {}
    axes = {
        "by_leader_life": lambda r: (f"leaderlife{r['own_leader_life']}"
                                     if r.get("own_leader_life") else None),
        "by_leader_colors": lambda r: ("+".join(r["own_leader_colors"])
                                       if r.get("own_leader_colors") else None),
        "by_leader_n_colors": lambda r: (f"{r['own_leader_n_colors']}color"
                                         if r.get("own_leader_n_colors") else None),
        "by_deck_rate": lambda r: (_rate_band(r["deck_rate"]) if r.get("deck_rate") is not None else None),
    }
    for name, key in axes.items():
        groups = collections.defaultdict(list)
        for r in recs:
            k = key(r)
            if k:
                groups[k].append(r)
        out[name] = {k: _group_block(v, min_n) for k, v in sorted(groups.items())
                     if len([x for x in v if x["kind"] == "d"]) >= min_n}
    # 注入テンプレートは「そのテンプレートを含むか」の重複あり分類
    tpl = {}
    for t in DECK_TEMPLATES:
        sub = [r for r in recs if t in (r.get("deck_templates") or ())]
        if len([x for x in sub if x["kind"] == "d"]) >= min_n:
            tpl[t] = _group_block(sub, min_n)
    out["by_deck_template"] = tpl
    # リーダー個別（出現数の多い上位）
    cnt = collections.Counter(r["own_leader"] for r in recs
                              if r["kind"] == "d" and r.get("own_leader"))
    top = [cid for cid, n in cnt.most_common(leader_top) if n >= min_n]
    out["by_leader"] = {cid: dict(_group_block([r for r in recs if r.get("own_leader") == cid], min_n),
                                  rows_d=cnt[cid]) for cid in top}
    # peak_life の散らばり（**しきい値が共通か個別かの答え**）
    def spread(d):
        peaks = [v["defense"].get("peak_life") for v in d.values()
                 if v["defense"].get("peak_life")]
        spans = [v["defense"].get("span") for v in d.values() if v["defense"].get("span")]
        return {"groups": len(d), "peaks": collections.Counter(peaks),
                "peak_dist": {str(k): v for k, v in sorted(collections.Counter(peaks).items())},
                "span_mean": (float(np.mean(spans)) if spans else None),
                "span_max": (float(max(spans)) if spans else None)}
    out["peak_spread"] = {k: spread(out[k]) for k in
                          ("by_leader", "by_leader_life", "by_leader_colors", "by_deck_template")
                          if out.get(k)}
    for k in out["peak_spread"]:
        out["peak_spread"][k].pop("peaks", None)
    return out


def _extra(dd, n):
    """race_state の状態列に加えて、forward に要る scalars／card_idx／tokens を丸ごと持つ。"""
    sc, tk = _race_extra(dd, n)
    return {"race_sc": sc, "race_tk": tk,
            "deck_kinds": (np.asarray(dd["deck_kinds"])[:n] if "deck_kinds" in dd.files else None),
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
    lead = Leaders(cards.db)
    lmap, meta_miss = game_leaders(dirs)
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
        ld = lmap.get(seed, (None, None))

        def strat(w, i):
            """行 i（席 w）の層別の軸（自席リーダーの属性＋手番側デッキの型）。"""
            out = dict(deck_info(str(ex["deck_kinds"][i])) if ex["deck_kinds"] is not None else {})
            cid = ld[w] if w in (0, 1) else None
            out["own_leader"] = cid
            out["opp_leader"] = ld[1 - w] if w in (0, 1) else None
            inf = lead.info(cid) if cid else None
            if inf:
                out.update(own_leader_life=inf["life"], own_leader_colors=inf["colors"],
                           own_leader_power=inf["power"], own_leader_n_colors=inf["n_colors"])
            return out
        for n, i in enumerate(idx):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) != 0 or (w, t) in seen_own:
                    continue
                seen_own.add((w, t))
                st = _state(ex["race_sc"][i], ex["race_tk"][i])
                rec = {"kind": "a", "turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]),
                       **st, **strat(w, i)}
                pend.append(("a", rec, ex["sc"][i], ex["ci"][i], ex["tok"][i]))
            else:
                if (w, t) in seen_opp:
                    continue
                seen_opp.add((w, t))
                rec = {"kind": "d", "turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]),
                       "my_life": int(round(float(ex["race_sc"][i][0]))),
                       "my_hand": int(round(float(ex["race_sc"][i][6]))), **strat(w, i)}
                pend.append(("d", rec, ex["sc"][i], ex["ci"][i], ex["tok"][i]))
            if len(pend) >= bs:
                flush()
    flush()
    return att, dfn, games, meta_miss, len(lmap)


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
    st = out.get("strat") or {}
    if st:
        print(f"--- 層別（リーダー／デッキ・meta_games の seed {out['meta_games_seeds']}・"
              f"sidecar 無しの dir {out['meta_games_missing_dirs']}）---")
        for name in ("by_leader_life", "by_leader_n_colors", "by_leader_colors",
                     "by_deck_rate", "by_deck_template", "by_leader"):
            blocks = st.get(name) or {}
            if not blocks:
                continue
            print(f"{name}（{len(blocks)} 群）")
            for k, v in blocks.items():
                dv = v["defense"]; av = v["attack_both_lethal"]
                curve = "  ".join(f"L{l}:{f(dv['by_life'][f'life{l}']['v_guard_minus_take'])}"
                                  for l in LIVES if f"life{l}" in dv["by_life"])
                print(f"  {str(k)[:22]:22} n{dv['n']:6d} peak L{dv.get('peak_life')} "
                      f"span {f(dv.get('span'))} | {curve} | 両方の圏 n{av.get('n', 0)} "
                      f"Vf-Vb {f(av.get('v_face_minus_board'))} amf {f(av.get('argmax_face'))} "
                      f"pf {f(av.get('played_face'))}")
        for name, sp in (st.get("peak_spread") or {}).items():
            print(f"peak_spread/{name}: 群 {sp['groups']} peak の分布 {sp['peak_dist']} "
                  f"span 平均 {f(sp['span_mean'])} 最大 {f(sp['span_max'])}")
    print(f"  {out['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="方針ヘッド付きの NRel npz")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed%%N==0 の対局だけ（0＝全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--leader-top", type=int, default=16,
                    help="個別に出すリーダーの数（相手ターンの行数の多い順）")
    ap.add_argument("--min-n", type=int, default=60,
                    help="群を出す最小の相手ターン行数（ライフ曲線の山を出す最小の n も同じ）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    net = NL.NRelNet.load(args.net, (stats, ab, abm, pwr, isl))
    if not net.plan:
        raise SystemExit("このネットに方針ヘッド（plan_*）が無い")
    from opcg_sim.learned.vocab import load_db
    rt = NR.RelTable(NR.profile_table(load_db(), vocab))
    att, dfn, games, meta_miss, n_meta = collect(net, rt, args.src, args.holdout_mod,
                                                 args.limit_games)
    out = {"net": os.path.abspath(args.net), "src": [os.path.abspath(s) for s in args.src],
           "holdout_mod": args.holdout_mod, "games": games,
           "meta_games_missing_dirs": meta_miss, "meta_games_seeds": n_meta,
           "attack": _attack_block(att), "defense": _defense_block(dfn),
           "strat": strat_blocks(att + dfn, args.leader_top, args.min_n),
           "strat_params": {"leader_top": args.leader_top, "min_n": args.min_n},
           "seconds": round(time.time() - t0, 1)}
    _print(out)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
