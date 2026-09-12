"""終盤の規則——ライフレースの状態に対して打ち方が合っているか（自己対戦の記録だけで測る）。

ユーザの終盤の考え方（2026-09-12）:
  ① 自分の攻撃回数が相手のライフより多ければリーサルを組む。
  ② それでも通らない（相手の手札・ブロッカー等）かつ次の相手の攻撃で自分が死ぬなら、除去・
     ブロッカー・キャラ攻撃でゲームを伸ばす。負け濃厚なら最初から①で突っ込むこともある。
守りは「手札とライフの交換レート」の管理＝序盤は受けて後半に守るデッキもあれば逆もある。

**状態**（自席ターンの最初の main 行＝そのターンの判断が見た盤面・符号化 v13/v14 から復元）:
  my_life／opp_life／my_hand／opp_hand … scalars[0,1,6,7]
  my_atk   … 今ターン攻撃できる数＝自リーダー（can_attack_now）＋自場の can_attack_now なキャラ
  opp_atk  … 次の相手ターンに殴ってくる数＝相手リーダー＋相手場のキャラ全部（相手は refresh する）
  my_blk／opp_blk … 場のブロッカー（is_blocker_active）
  lethal_margin … my_atk − opp_blk − (opp_life + 1)   ≥0 で「リーサル圏」（ライフ+1 回通せば勝ち）
  threat_margin … opp_atk − my_blk − (my_life + 1)    ≥0 で「被リーサル圏」
  相手の手札はカウンターの見込み（1 枚 ≈ 攻撃 1 回を止める）として層別に使う（中身は見えない）。

**行動**（そのターンの main 決定・`plan_drift.py` と同じ分類）:
  n_face（リーダー攻撃）・n_board（キャラ攻撃＋除去効果）・n_play・blocker_played（ブロッカーを出した）
  attempted … n_face ≥ min(my_atk, opp_life + 1)＝手数の許す限りリーダーを殴った（リーサルを狙った）
  defended  … n_board ≥ 1 or blocker_played（盤面を触った／守りを足した）
  won_now   … その自席ターンで対局が終わり自分が勝った（リーサルが通った）
  died_next … 次の相手ターンで対局が終わり自分が負けた

**守り**（相手ターンごと）: 相手のリーダー攻撃の回数と、その相手ターンで減った自分のライフから
  guard_rate = 1 − 減ったライフ / リーダー攻撃回数（受けた割合の裏）。ターン帯・自分のライフで層別。

出すもの:
  1. 状態の分布（lethal_margin × threat_margin の表）
  2. リーサル圏: attempted 率・won_now 率・勝率（attempted 別）・相手手札で層別。**見逃し**＝圏内で
     attempted でない割合
  3. 被リーサル圏（リーサル圏でない）: defended 率・died_next 率・勝率（defended 別）・自分の手札で層別。
     **無謀**＝圏内で defended せず face だけ
  4. 両方の圏（純粋なレース）: attempted／defended の内訳と勝率
  5. 勝者／敗者・ターン帯で 2〜4 を比べる
  6. 守り: guard_rate をターン帯 × 自分のライフで

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/race_state.py --in ~/n32_wave/w01/n_records/* --out ~/rs_w32.json
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

from plan_drift import _Cards, _cls, _iter_games, _prop_ci  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402

_ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
_POL_COLS = ("pol_sig", "pol_cid", "pol_tcid")
# tokens の列（n_rel_feat.S_COLS）・枠（0 自L・1 相L・2〜6 自場・7〜11 相場）
_C_PWR, _C_REST, _C_CAN, _C_BLK, _C_CHAR = 0, 3, 5, 6, 18
_TOK_COLS = (_C_PWR, _C_REST, _C_CAN, _C_BLK, _C_CHAR)
_T_PWR, _T_REST, _T_CAN, _T_BLK, _T_CHAR = range(len(_TOK_COLS))
_SLOT_OWN = slice(2, 7); _SLOT_OPP = slice(7, 12)
#: `power_now` の目盛り（`n_rel_feat` は 10000 で割って入れる）
_PWR_SCALE = 10000.0


def _turn_band(t):
    return "t1-4" if t <= 4 else "t5-8" if t <= 8 else "t9+"


def _hand_band(h):
    return "h0-1" if h <= 1 else "h2-3" if h <= 3 else "h4+"


def _margin_band(m):
    return "<=-2" if m <= -2 else "-1" if m == -1 else "0" if m == 0 else "+1" if m == 1 else ">=+2"


def _pwr_band(p):
    """飛んでくる攻撃の超過パワー（攻撃側 − 自リーダー）の帯（単位 1000）。

    **カウンター 1 枚は 1000 か 2000** なので、この帯がそのまま「守るのに何枚要るか」に対応する
    （<=0＝守らなくても耐える・1000〜2000＝1 枚・3000〜4000＝2 枚・5000+＝3 枚以上）。
    """
    if p is None:
        return None
    k = int(round(p / 1000.0))
    return "p<=0" if k <= 0 else "p1-2k" if k <= 2 else "p3-4k" if k <= 4 else "p5k+"


def _extra(dd, n):
    """scalars の先頭 14（ライフ・ドン・手札・場・ターン・リーダーパワー）と tokens の 12 枠 × 5 列。"""
    sc = np.asarray(dd["scalars"])[:n, :14].astype(np.float32)
    tk = np.asarray(dd["tokens"])[:n, :12][:, :, list(_TOK_COLS)].astype(np.float32)
    return sc, tk


def _incoming(tk):
    """相手ターン中の自分の行 → 飛んでくる攻撃の超過パワー（最大の攻撃側 − 自リーダー）。

    `power_now` は**その行が誰の手番かで評価が切り替わる**（`n_rel_feat` の v14 の注記）ので、
    相手ターン中の自分の行では「相手の攻撃パワー」と「自分の守りのパワー」がそのまま入っている。
    相手の枠（リーダー＋場のキャラ）の最大パワーから自リーダーのパワーを引く。攻撃側が居ない
    （枠が全て空）なら None。
    """
    opp = tk[_SLOT_OPP]
    cand = [float(tk[1, _T_PWR])]                      # 相手リーダー
    ch = opp[:, _T_CHAR] > 0.5
    if ch.any():
        cand.append(float(opp[ch, _T_PWR].max()))
    top = max(cand) * _PWR_SCALE
    mine = float(tk[0, _T_PWR]) * _PWR_SCALE
    if top <= 0.0:
        return None
    return top - mine


def _state(sc, tk):
    """自席ターン開始の main 行 → レースの状態。tk は [12, 4]（rest, can, blk, char）。"""
    own = tk[_SLOT_OWN]; opp = tk[_SLOT_OPP]
    own_ch = own[:, _T_CHAR] > 0.5; opp_ch = opp[:, _T_CHAR] > 0.5
    my_atk = int(tk[0, _T_CAN] > 0.5) + int((own_ch & (own[:, _T_CAN] > 0.5)).sum())
    opp_atk = 1 + int(opp_ch.sum())
    my_blk = int((own_ch & (own[:, _T_BLK] > 0.5)).sum())
    opp_blk = int((opp_ch & (opp[:, _T_BLK] > 0.5)).sum())
    my_life, opp_life = int(round(sc[0])), int(round(sc[1]))
    my_hand, opp_hand = int(round(sc[6])), int(round(sc[7]))
    return {"my_life": my_life, "opp_life": opp_life, "my_hand": my_hand, "opp_hand": opp_hand,
            "my_atk": my_atk, "opp_atk": opp_atk, "my_blk": my_blk, "opp_blk": opp_blk,
            "lethal_margin": my_atk - opp_blk - (opp_life + 1),
            "threat_margin": opp_atk - my_blk - (my_life + 1)}


def analyze(dirs, limit_games=0):
    t0 = time.time()
    cards = _Cards(D.load_db())
    db = cards.db
    n_rows = 0

    def is_blocker(cid):
        m = db.get_card(cid) if cid else None
        return bool(m is not None and "ブロッカー" in (getattr(m, "keywords", ()) or ()))

    turn_recs = []                      # 自席ターン 1 つ = 1 レコード
    guard_recs = []                     # 相手ターン 1 つ = 1 レコード
    games = 0
    for rows, pol, (sc, tk), L, ptr, idx in _iter_games(dirs, _ROW_COLS, _POL_COLS, _extra):
        games += 1
        if limit_games and games > limit_games:
            break
        n_rows += len(idx)
        u2c = {}
        for i in idx:
            k = int(L[i])
            for j in range(ptr[i], ptr[i] + k):
                sj = json.loads(pol["pol_sig"][j])
                if sj[1] and pol["pol_cid"][j]:
                    u2c[sj[1]] = str(pol["pol_cid"][j])
                if sj[2] and pol["pol_tcid"][j]:
                    u2c[sj[2][0]] = str(pol["pol_tcid"][j])
        # (who, turn) → 状態（最初の main 行）と行動の集計
        first = {}; acts = {}; zs = {}
        # (who, 相手ターン) → 飛んでくる攻撃の超過パワー（自分の**最初の**行で測る・§D）
        incoming = {}
        last_turn = max(int(rows["turn"][i]) for i in idx)
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            zs[w] = float(rows["z"][i])
            if t >= 1 and (t % 2 == 1) != (w == 0) and (w, t) not in incoming:
                incoming[(w, t)] = _incoming(tk[i])      # 相手ターン中の自分の行（守りの層別用）
            if t < 1 or (t % 2 == 1) != (w == 0) or int(rows["kind"][i]) != 0:
                continue
            key = (w, t)
            if key not in first:
                first[key] = _state(sc[i], tk[i])
                acts[key] = collections.Counter()
            sj = json.loads(rows["sig"][i])
            cl = _cls(sj, u2c, cards)
            if cl in ("face", "board", "develop"):
                acts[key][cl] += 1
            if sj[0] == "PLAY" and is_blocker(u2c.get(sj[1])):
                acts[key]["blocker"] += 1
        # 対局の終わり: 最後の行の手番と勝者
        winner = 0 if zs.get(0, 0.0) > 0 else 1
        for (w, t), st in first.items():
            a = acts[(w, t)]
            z = zs.get(w, 0.0)
            ended_here = (t == last_turn)
            won_now = ended_here and winner == w
            died_next = (last_turn == t + 1) and winner != w
            rec = dict(st)
            rec.update({"who": w, "turn": t, "z": z, "n_face": a["face"], "n_board": a["board"],
                        "n_play": a["develop"], "blocker": a["blocker"],
                        "attempted": a["face"] >= max(1, min(st["my_atk"], st["opp_life"] + 1)),
                        "defended": (a["board"] >= 1 or a["blocker"] >= 1),
                        "won_now": won_now, "died_next": died_next, "ended_here": ended_here})
            turn_recs.append(rec)
        # 守り: 相手ターン t（相手の自席ターン）でのリーダー攻撃回数と、自分のライフの減り
        # （自分の次の自席ターン開始の my_life と、前の自席ターン開始の my_life の差。ライフ回復は無視）
        for w in (0, 1):
            ts = sorted(t for (ww, t) in first if ww == w)
            for a_t, b_t in zip(ts, ts[1:]):
                opp_key = (1 - w, a_t + 1)
                if opp_key not in acts:
                    continue
                n_att = acts[opp_key]["face"]
                lost = first[(w, a_t)]["my_life"] - first[(w, b_t)]["my_life"]
                if n_att <= 0:
                    continue
                guard_recs.append({"who": w, "turn": a_t + 1, "z": zs.get(w, 0.0), "attacks": n_att,
                                   "lost": max(0, lost), "life_before": first[(w, a_t)]["my_life"],
                                   "hand_before": first[(w, a_t)]["my_hand"],
                                   "atk_over": incoming.get((w, a_t + 1))})
    print(f"行 {n_rows}・対局 {games}（{time.time()-t0:.0f}s）", flush=True)
    return _aggregate(turn_recs, guard_recs, games, t0)


def _rate(recs, key):
    return _prop_ci(sum(1 for r in recs if r[key]), len(recs))


def _winrate(recs):
    return _prop_ci(sum(1 for r in recs if r["z"] > 0), len(recs))


def _zone_block(recs, split_key, split_fn):
    """圏内のレコード → attempted／defended／結果と、層別。"""
    out = {"turns": len(recs)}
    if not recs:
        return out
    out["attempted"] = _rate(recs, "attempted")
    out["defended"] = _rate(recs, "defended")
    out["won_now"] = _rate(recs, "won_now")
    out["died_next"] = _rate(recs, "died_next")
    out["winrate"] = _winrate(recs)
    for name, cond in (("attempted", lambda r: r["attempted"]), ("not_attempted", lambda r: not r["attempted"]),
                       ("defended", lambda r: r["defended"]), ("not_defended", lambda r: not r["defended"]),
                       ("face_only", lambda r: r["n_face"] > 0 and not r["defended"])):
        sub = [r for r in recs if cond(r)]
        out[f"winrate_{name}"] = _winrate(sub)
        out[f"won_now_{name}"] = _rate(sub, "won_now") if sub else {"n": 0}
        out[f"died_next_{name}"] = _rate(sub, "died_next") if sub else {"n": 0}
    out[f"by_{split_key}"] = {}
    for band in ("h0-1", "h2-3", "h4+"):
        sub = [r for r in recs if split_fn(r) == band]
        if sub:
            out[f"by_{split_key}"][band] = {"turns": len(sub), "attempted": _rate(sub, "attempted"),
                                            "defended": _rate(sub, "defended"), "won_now": _rate(sub, "won_now"),
                                            "died_next": _rate(sub, "died_next"), "winrate": _winrate(sub)}
    out["by_turn"] = {}
    for band in ("t1-4", "t5-8", "t9+"):
        sub = [r for r in recs if _turn_band(r["turn"]) == band]
        if sub:
            out["by_turn"][band] = {"turns": len(sub), "attempted": _rate(sub, "attempted"),
                                    "defended": _rate(sub, "defended"), "winrate": _winrate(sub)}
    out["by_z"] = {"win": {"turns": sum(1 for r in recs if r["z"] > 0),
                           "attempted": _rate([r for r in recs if r["z"] > 0], "attempted"),
                           "defended": _rate([r for r in recs if r["z"] > 0], "defended")},
                   "lose": {"turns": sum(1 for r in recs if r["z"] < 0),
                            "attempted": _rate([r for r in recs if r["z"] < 0], "attempted"),
                            "defended": _rate([r for r in recs if r["z"] < 0], "defended")}}
    return out


def _aggregate(turn_recs, guard_recs, games, t0):
    out = {"games": games, "own_turns": len(turn_recs)}
    # 1. 状態の分布
    dist = collections.Counter((_margin_band(r["lethal_margin"]), _margin_band(r["threat_margin"])) for r in turn_recs)
    out["state_dist"] = {f"L{a}|T{b}": v for (a, b), v in sorted(dist.items())}
    lethal = [r for r in turn_recs if r["lethal_margin"] >= 0]
    threat = [r for r in turn_recs if r["threat_margin"] >= 0]
    both = [r for r in lethal if r["threat_margin"] >= 0]
    lethal_only = [r for r in lethal if r["threat_margin"] < 0]
    threat_only = [r for r in threat if r["lethal_margin"] < 0]
    neither = [r for r in turn_recs if r["lethal_margin"] < 0 and r["threat_margin"] < 0]
    out["lethal_only"] = _zone_block(lethal_only, "opp_hand", lambda r: _hand_band(r["opp_hand"]))
    out["threat_only"] = _zone_block(threat_only, "my_hand", lambda r: _hand_band(r["my_hand"]))
    out["both"] = _zone_block(both, "opp_hand", lambda r: _hand_band(r["opp_hand"]))
    out["neither"] = {"turns": len(neither), "attempted": _rate(neither, "attempted") if neither else {"n": 0},
                      "defended": _rate(neither, "defended") if neither else {"n": 0},
                      "winrate": _winrate(neither) if neither else {"n": 0}}
    # リーサル圏の margin 別（+0 / +1 / ≥+2）
    out["lethal_by_margin"] = {}
    for band in ("0", "+1", ">=+2"):
        sub = [r for r in lethal if _margin_band(r["lethal_margin"]) == band]
        if sub:
            out["lethal_by_margin"][band] = {"turns": len(sub), "attempted": _rate(sub, "attempted"),
                                             "won_now": _rate(sub, "won_now"), "winrate": _winrate(sub)}
    # 見逃し・無謀の頻度（ターン単位・勝者／敗者で比べる）
    out["miss_rate"] = _prop_ci(sum(1 for r in lethal if not r["attempted"]), len(lethal))
    out["reckless_rate"] = _prop_ci(sum(1 for r in threat_only if r["n_face"] > 0 and not r["defended"]),
                                    len(threat_only))
    out["miss_by_z"] = {"win": _prop_ci(sum(1 for r in lethal if r["z"] > 0 and not r["attempted"]),
                                        sum(1 for r in lethal if r["z"] > 0)),
                        "lose": _prop_ci(sum(1 for r in lethal if r["z"] < 0 and not r["attempted"]),
                                         sum(1 for r in lethal if r["z"] < 0))}
    # 6. 守り
    g = {"opp_turns_with_leader_attacks": len(guard_recs)}
    if guard_recs:
        att = sum(r["attacks"] for r in guard_recs); lost = sum(r["lost"] for r in guard_recs)
        g["guard_rate"] = 1.0 - lost / att
        g["by_turn"] = {}
        for band in ("t1-4", "t5-8", "t9+"):
            sub = [r for r in guard_recs if _turn_band(r["turn"]) == band]
            if sub:
                a = sum(r["attacks"] for r in sub); l_ = sum(r["lost"] for r in sub)
                g["by_turn"][band] = {"attacks": a, "guard_rate": 1.0 - l_ / a}
        g["by_life"] = {}
        for lb in (1, 2, 3, 4, 5):
            sub = [r for r in guard_recs if r["life_before"] == lb]
            if sub:
                a = sum(r["attacks"] for r in sub); l_ = sum(r["lost"] for r in sub)
                g["by_life"][f"life{lb}"] = {"attacks": a, "guard_rate": 1.0 - l_ / a}
        g["by_z"] = {}
        for name, cond in (("win", lambda r: r["z"] > 0), ("lose", lambda r: r["z"] < 0)):
            sub = [r for r in guard_recs if cond(r)]
            if sub:
                a = sum(r["attacks"] for r in sub); l_ = sum(r["lost"] for r in sub)
                g["by_z"][name] = {"attacks": a, "guard_rate": 1.0 - l_ / a}
        # **飛んでくる攻撃の超過パワー別**（§D・「安い攻撃は受けて高い攻撃を守る」の検査）。
        # 帯はカウンター 1 枚（1000〜2000）を単位にしている＝守るのに要る枚数に対応する。
        g["by_atk_power"] = {}
        for band in ("p<=0", "p1-2k", "p3-4k", "p5k+"):
            sub = [r for r in guard_recs if _pwr_band(r.get("atk_over")) == band]
            if sub:
                a = sum(r["attacks"] for r in sub); l_ = sum(r["lost"] for r in sub)
                g["by_atk_power"][band] = {"attacks": a, "turns": len(sub),
                                           "guard_rate": 1.0 - l_ / a,
                                           "mean_over": float(np.mean([r["atk_over"] for r in sub])),
                                           "winrate": float(np.mean([r["z"] > 0 for r in sub]))}
        g["atk_power_unknown"] = sum(1 for r in guard_recs if _pwr_band(r.get("atk_over")) is None)
        # 超過パワー × 自分のライフ（**交換レートの管理が在るか**＝安い攻撃をライフで受け、
        # 高い攻撃を手札で止める、がライフ残に応じて動いているか）
        g["by_atk_power_life"] = {}
        for band in ("p<=0", "p1-2k", "p3-4k", "p5k+"):
            row = {}
            for lb in (1, 2, 3, 4, 5):
                sub = [r for r in guard_recs
                       if _pwr_band(r.get("atk_over")) == band and r["life_before"] == lb]
                if sub:
                    a = sum(r["attacks"] for r in sub); l_ = sum(r["lost"] for r in sub)
                    row[f"life{lb}"] = {"attacks": a, "guard_rate": 1.0 - l_ / a}
            if row:
                g["by_atk_power_life"][band] = row
    out["guard"] = g
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _p(p):
    return f"{p['p']:.3f} [{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] n {p['n']}" if p.get("n") else "(0)"


def _print(out):
    print(f"対局 {out['games']}・自席ターン {out['own_turns']}")
    print("状態の分布（L=lethal_margin・T=threat_margin）:", out["state_dist"])
    for zone in ("lethal_only", "threat_only", "both"):
        b = out[zone]
        print(f"{zone}: {b['turns']} ターン")
        if b["turns"]:
            print(f"  attempted {_p(b['attempted'])}  defended {_p(b['defended'])}  won_now {_p(b['won_now'])}"
                  f"  died_next {_p(b['died_next'])}  winrate {_p(b['winrate'])}")
            for nm in ("attempted", "not_attempted", "defended", "not_defended", "face_only"):
                print(f"    {nm:14} winrate {_p(b['winrate_' + nm])}  won_now {_p(b['won_now_' + nm])}"
                      f"  died_next {_p(b['died_next_' + nm])}")
            for k, v in b.items():
                if k.startswith("by_") and isinstance(v, dict):
                    for band, d in v.items():
                        print(f"    {k}/{band:6} turns {d['turns']:5d}  attempted {_p(d['attempted'])}"
                              f"  defended {_p(d['defended'])}" + (f"  winrate {_p(d['winrate'])}" if "winrate" in d else ""))
    print(f"neither: {out['neither']['turns']} ターン  attempted {_p(out['neither']['attempted'])}"
          f"  defended {_p(out['neither']['defended'])}")
    print("lethal_by_margin:", {k: (v["turns"], round(v["attempted"]["p"], 3), round(v["won_now"]["p"], 3))
                                for k, v in out["lethal_by_margin"].items()})
    print(f"見逃し（リーサル圏で attempted でない）{_p(out['miss_rate'])}  win {_p(out['miss_by_z']['win'])}"
          f"  lose {_p(out['miss_by_z']['lose'])}")
    print(f"無謀（被リーサル圏で face だけ）{_p(out['reckless_rate'])}")
    g = out["guard"]
    if g.get("guard_rate") is not None:
        def fmt(d):
            return "{" + ", ".join(f"{k}: {v['guard_rate']:.3f}" for k, v in d.items()) + "}"
        print(f"守り guard_rate {g['guard_rate']:.3f}（相手ターン {g['opp_turns_with_leader_attacks']}）"
              f"  by_turn {fmt(g['by_turn'])}  by_life {fmt(g['by_life'])}  by_z {fmt(g['by_z'])}")
        if g.get("by_atk_power"):
            print(f"  超過パワー別（不明 {g['atk_power_unknown']}）: " + "  ".join(
                f"{k}: n{v['turns']} guard {v['guard_rate']:.3f} 平均 {v['mean_over']:+.0f} wr {v['winrate']:.3f}"
                for k, v in g["by_atk_power"].items()))
            for k, row in g["by_atk_power_life"].items():
                print(f"    {k:6} × ライフ: " + "  ".join(
                    f"{lb}:{v['guard_rate']:.3f}(n{v['attacks']})" for lb, v in row.items()))
    print(f"  {out['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ（v13/v14 の波）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = analyze(args.src, args.limit_games)
    out["src"] = [os.path.abspath(s) for s in args.src]
    _print(out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
