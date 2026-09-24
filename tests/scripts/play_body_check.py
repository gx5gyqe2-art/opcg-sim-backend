"""**出す手の値段は両端で外れているか**——出した体を追い、値段の部品と実際に積んだものを費用帯ごとに並べる
（T157・2026-09-24・読み取り専用・記録だけ・回帰しない）。

T18-why2（`2026-09-24_t18_why2.md`）は 6 局の影の判定から「安い体は札 1 枚ぶんにしか見えず（出す価格が負）、
大きい体は札 3 枚ぶんに見える」と読み、**直す場所は出す手の価格（#43）**とした。ただしその「誤り」は
(1) 安い札には別の使い道がほとんど無い（原則からの推論・未測定）、(2) 探索が大きい体の登場を否定する
（探索が正しいとは限らない）の 2 つに立っていた（ユーザ質問 2026-09-24「a は何を基準にそう判断したの？」）。
本器はその 2 つを**記録で**確かめる。

## 並べるもの（出す手の行ごと・キャラの登場だけ）

| 値段の部品（`price_realised` と同じ式） | 記録で測るもの |
|---|---|
| **手札 1 枚の代金 `μ`**（全札一律） | **その札の別の使い道**＝出さずに持っていたときの守りの価値 `ΔG_guard`（T67・`hand_plan.card_deltas`）と、その札のカウンター値 |
| **体の価値 `ν`**（式・出した瞬間の盤面） | **体を追った在庫**＝その体が生きている間に打った攻撃の実現の和（T50 `nu_stock` の作法・枠 × `card_idx` で追う）・生きて迎えた自席ターン数・攻撃回数・終局前に場から消えた割合 |
| ドンの機会費用・登場時効果 | （並べるだけ） |

## 検算の予告（事前登録・測る前に書く）

- **(a) の読みが正しいなら**: 費用 1〜2 の札の `ΔG_guard` は `μ` を明確に下回り（別の使い道が薄い）、
  「在庫 ÷ 式の `ν`」の比は費用帯をまたいで**安い側で高く・高い側で低く**傾く（安い体は稼ぎ以上に安く、
  大きい体は稼ぎ以上に高く値付けされている）。
- **ユーザの読み（理論が正しい）なら**: 費用 1〜2 の札の `ΔG_guard` は `μ` と同程度（安い札ほどカウンターが厚い）、
  比は費用帯をまたいで平ら。
- **比の水準そのもの**は 1 を下回ってよい——在庫は攻撃しか数えない（ブロック・相手の除去を吸った分・抑止は
  体に帰属できない・T50 で 0.5〜0.77）。**読むのは帯をまたいだ傾きだけ**。

**限界**: 体の同一性は枠 × `card_idx`（記録に uuid が無い・P8）。登場した枠は「次の判断点で同じ札が新しく居る枠」。
相手が除去に払った札・ドンは体に帰属できない（大きい体ほど除去を吸うなら、在庫は大きい体を低く見せる向きに偏る）。

使い方: `OPCG_LOG_SILENT=1 python tests/scripts/play_body_check.py --in <n_records>... [--out x.json]`
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
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import guard_afford as GA  # noqa: E402
import hand_guard as HG  # noqa: E402
import hand_plan as HP  # noqa: E402
import price_realised as PR  # noqa: E402
from price_realised import don_stock, state_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, move_family  # noqa: E402
from theory_order import (MU, S_IS_CHAR, SC_MY_DON, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE,  # noqa: E402
                          SLOT_OWN_FIELD, THETA, play_cost_term, play_value, score_candidate, slot_power)

#: 費用帯（印字のコスト）。境目は規則の数字だけ（1〜2＝序盤に出す札・10 は上限）
COST_BANDS = (("c1_2", 1, 2), ("c3_4", 3, 4), ("c5_6", 5, 6), ("c7_10", 7, 10))
#: 式の `ν` を手札の枚数（`μ` 単位）で切る帯——T18-why2 の「1 枚ぶん」「3 枚ぶん」と同じ言葉で読むため
NU_BANDS = (("nu_lt_1_5", 0.0, 1.5), ("nu_1_5_2_5", 1.5, 2.5), ("nu_ge_2_5", 2.5, 1e9))


def cost_band(cost):
    c = int(round(float(cost)))
    for name, lo, hi in COST_BANDS:
        if lo <= c <= hi:
            return name
    return "c0"


def nu_band(nu, mu=MU):
    x = float(nu) / float(mu)
    for name, lo, hi in NU_BANDS:
        if lo <= x < hi:
            return name
    return NU_BANDS[-1][0]


def landing_slot(tok_before, ci_before, tok_after, ci_after, cid_idx):
    """出した体が入った枠——後の盤面で `cid_idx` のキャラが居て、前の盤面ではその枠に同じ札が居なかった枠。
    見つからなければ `None`（登場時に場を離れた・読み違い）。候補が複数なら最初の枠。"""
    ci_b, ci_a = np.asarray(ci_before), np.asarray(ci_after)
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok_after[s, S_IS_CHAR]) <= 0.5 or int(ci_a[s]) != int(cid_idx):
            continue
        if float(tok_before[s, S_IS_CHAR]) > 0.5 and int(ci_b[s]) == int(cid_idx):
            continue
        return s
    return None


def hand_card_value(sc, tok, ci_row, idx2cid, cards, cid, deck=None):
    """出す直前の手札で、出す札の `ΔG_guard`（持っていたら守りに使える価値）・`ΔH_play`・カウンター値。
    同じ札が 2 枚あれば最初の 1 枚。手札に見つからなければ `None`。"""
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
    caps = HP.caps_of(float(sc[SC_MY_DON]), don_stock(sc, tok, "me"), r_turns=r)
    xs = HG.incoming(tok)
    take = HG.take_cost_of(float(sc[SC_MY_LIFE]))
    items = HP.apply_inflow(HP.hand_items(tok, ci_row, idx2cid, cards, olp, r), deck, xs, take, cards, olp, r,
                            field=HP.own_field_ids(ci_row, idx2cid),
                            st_base=HP.state_of_row(sc, tok, ci_row, idx2cid, cards))
    k = next((q for q, it in enumerate(items) if it["cid"] == str(cid)), None)
    if k is None:
        return None
    rest, card = items[:k] + items[k + 1:], items[k]
    d = HP.card_deltas(rest, card, caps, xs, take)
    # **持っておいて後のターンに出す**価値＝今のターンの枠を 0 にした計画での `ΔH`（今は出さない前提）
    caps_later = [0] + list(caps[1:])
    dh_later = HP.delta_h([(it["cost"], it["v"]) for it in rest], (card["cost"], card["v"]), caps_later)
    return {"dg": float(d["dg"]), "dh": float(d["dh"]), "dh_later": float(dh_later),
            "alt": float(max(d["dg"], dh_later)), "counter": float(d["counter"]),
            "counter_card": bool(d["counter_card"])}


def follow_body(turn_start, ts, j, slot, cid_idx, atk_real):
    """登場したターン（`ts[j]`）から、枠 `slot` に同じ札が居る間の攻撃の実現を足す（`nu_stock` の作法）。
    戻り値: 在庫・攻撃回数・生きて迎えた自席ターン数（登場ターンを含む）・終局前に消えたか。"""
    stock, n_atk, turns_alive, died = 0.0, 0, 0, False
    for kk in range(0, len(ts) - j):
        if kk > 0:
            _sc2, tok2, ci2 = turn_start[ts[j + kk]]
            if not (float(tok2[slot, S_IS_CHAR]) > 0.5 and int(ci2[slot]) == int(cid_idx)):
                died = True
                break
        turns_alive += 1
        got = atk_real.get((ts[j + kk], slot), [])
        stock += sum(got)
        n_atk += len(got)
    return stock, n_atk, turns_alive, died


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    from theory_bridge import _seat_decks          # 遅延 import（橋は本器を import しない）
    import search_price as SP
    import effect_value as EV
    cards = PL.Cards()
    vocab = GA._vocab()
    idx2cid = {i: c for c, i in vocab.items()}
    rec_decks = SP.record_decks(dirs) if EV.SEARCH_PRICE_MODE == "plan" else {}
    out = []
    stats = {"games": 0, "play_rows": 0, "no_slot": 0, "no_hand": 0, "silent": 0, "search_deck_ok": 0, "search_deck_bad": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        seed = int(rows["seed"][idx[0]])
        decks = _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats)
        by_seat = {}
        for n, i in enumerate(order):
            if int(rows["kind"][i]) == 0:
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {a: b for ns in by_seat.values() for a, b in zip(ns, ns[1:])}
        # 席ごとの自席ターン開始の盤面・枠ごとの攻撃の実現・登場の行
        turn_start, turn_seq, atk_real, plays = {}, {}, {}, []
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            if t not in turn_start.setdefault(w, {}):
                turn_start[w][t] = (sc, tok, np.asarray(ex["ci"][i]))
                turn_seq.setdefault(w, []).append(t)
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            fam = move_family(sig)
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue
            i2 = order[j]
            if fam == "attack":
                real = float(state_meas(ex["sc"][i2], ex["tok"][i2]) - state_meas(sc, tok))
                atk_real.setdefault(w, {}).setdefault((t, int(pol["pol_si"][b])), []).append(real)
            elif fam == "play":
                plays.append((n, i, i2, b, sig, w, t))
        for n, i, i2, b, sig, w, t in plays:
            cid = str(pol["pol_cid"][b]) or None
            info = cards.info(cid) if cid else None
            if info is None or info.get("event") or info.get("stage") or info.get("power") is None:
                continue
            stats["play_rows"] += 1
            sc, tok = ex["sc"][i], ex["tok"][i]
            ctx = PR.row_ctx(sc, tok, ex["ci"][i], idx2cid, cards, decks.get(w), theta, mu, theta_mode)
            th = ctx["theta"]
            v = score_candidate(sig, cid, None, ctx, cards, src_power=slot_power(tok, pol["pol_si"][b]),
                                tgt_power=None, don_k=pol["pol_k"][b])
            if v is None:
                stats["silent"] += 1
                continue
            cost = float(info.get("cost") or 0.0)
            nu_minus_mu = float(play_value(float(info["power"]), 0, ctx["opp_leader_power"], ctx["r_turns"], th, mu,
                                           is_blocker=info.get("blocker"), my_leader_power=ctx["my_leader_power"]))
            nu = nu_minus_mu + mu
            opp_cost = float(play_cost_term(ctx, cost, mu, th))
            effect = float(v) - nu_minus_mu + opp_cost
            hv = hand_card_value(sc, tok, ex["ci"][i], idx2cid, cards, cid, deck=decks.get(w))
            if hv is None:
                stats["no_hand"] += 1
            cid_idx = vocab.get(cid)
            slot = (landing_slot(tok, ex["ci"][i], ex["tok"][i2], ex["ci"][i2], cid_idx)
                    if cid_idx is not None else None)
            row = {"cid": cid, "cost": cost, "power": float(info["power"]), "turn": t,
                   "cost_band": cost_band(cost), "nu_band": nu_band(nu, mu),
                   "price": float(v), "nu": nu, "mu": float(mu), "opportunity": opp_cost, "effect": effect,
                   "dg": hv["dg"] if hv else None, "dh": hv["dh"] if hv else None,
                   "dh_later": hv["dh_later"] if hv else None, "alt": hv["alt"] if hv else None,
                   "counter": hv["counter"] if hv else None, "counter_card": hv["counter_card"] if hv else None,
                   "slot": slot}
            if slot is None:
                stats["no_slot"] += 1
            else:
                ts = turn_seq[w]
                jj = ts.index(t)
                ts_map = turn_start[w]
                stock, n_atk, alive, died = follow_body(ts_map, ts, jj, slot, cid_idx, atk_real.get(w, {}))
                row.update({"stock": stock, "n_atk": n_atk, "turns_alive": alive, "died": died,
                            "turns_left": len(ts) - jj})
            out.append(row)
    return out, stats


def _m(rs, key, nd=4):
    xs = [r[key] for r in rs if r.get(key) is not None]
    return round(float(np.mean(xs)), nd) if xs else None


def block(rs, mu=MU):
    tracked = [r for r in rs if r.get("slot") is not None]
    nu, stock = _m(tracked, "nu", 6), _m(tracked, "stock", 6)
    o = {"n": len(rs), "n_tracked": len(tracked),
         "price_cards": round(_m(rs, "price", 6) / mu, 3),
         "nu_cards": round(_m(rs, "nu", 6) / mu, 3),
         "opportunity_cards": round(_m(rs, "opportunity", 6) / mu, 3),
         "effect_cards": round(_m(rs, "effect", 6) / mu, 3),
         "neg_price_share": round(float(np.mean([r["price"] < 0 for r in rs])), 3),
         # 別の使い道（持っていたら守りに使える価値）対 代金 μ
         "guard_alt_cards": (round(_m(rs, "dg", 6) / mu, 3) if _m(rs, "dg") is not None else None),
         "play_plan_cards": (round(_m(rs, "dh", 6) / mu, 3) if _m(rs, "dh") is not None else None),
         # 持っておいて後のターンに出す価値・別の使い道の最大（守り／後で出す）
         "later_alt_cards": (round(_m(rs, "dh_later", 6) / mu, 3) if _m(rs, "dh_later") is not None else None),
         "alt_cards": (round(_m(rs, "alt", 6) / mu, 3) if _m(rs, "alt") is not None else None),
         "counter_mean": _m(rs, "counter", 0),
         "counter_card_share": _m(rs, "counter_card", 3),
         "guard_zero_share": (round(float(np.mean([r["dg"] <= 1e-9 for r in rs if r.get("dg") is not None])), 3)
                              if any(r.get("dg") is not None for r in rs) else None)}
    if tracked:
        o.update({"stock_cards": round(stock / mu, 3),
                  "stock_over_nu": round(stock / nu, 3) if nu else None,
                  "turns_alive": _m(tracked, "turns_alive", 3),
                  "turns_left": _m(tracked, "turns_left", 3),
                  "attacks": _m(tracked, "n_atk", 3),
                  "died_share": round(float(np.mean([r["died"] for r in tracked])), 3),
                  "real_per_attack_cards": (round(sum(r["stock"] for r in tracked) / max(1, sum(r["n_atk"] for r in tracked)) / mu, 3))})
    return o


def summarise(rows, mu=MU):
    out = {"all": block(rows, mu), "by_cost": {}, "by_nu": {}}
    for name, _lo, _hi in COST_BANDS:
        rs = [r for r in rows if r["cost_band"] == name]
        if rs:
            out["by_cost"][name] = block(rs, mu)
    for name, _lo, _hi in NU_BANDS:
        rs = [r for r in rows if r["nu_band"] == name]
        if rs:
            out["by_nu"][name] = block(rs, mu)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows, stats = collect(a.src, a.limit_games)
    res = {"stats": stats, "mu": MU, "summary": summarise(rows), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
