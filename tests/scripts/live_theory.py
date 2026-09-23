#!/usr/bin/env python3
"""**T140**（2026-09-23）: 理論を **生の局面**（`Game.encode` の直接呼び出し）から読む橋と、
「生 = 記録」の検算。

## 問い

T121〜T143 の理論の道具（`kappa_vector.state_of_row`／`theory_order.score_candidate` など）は
**すべて記録（npz ダンプ）の行から読む**——`sc`／`tok`／`ci` は生成時に一度 `game.encode()` した後
float16／int16 へ落として保存したもの。T18／T141（打ち回しへの介入）は**生成の外**で理論を使う
（打っている最中の対局に「この手は理論なら禁じる」と問う）ので、**記録を経由しない生の局面**から
同じ理論値が読めることを先に確かめておく必要がある。

## 式（新定数ゼロ・既存の道具の再利用のみ）

* **状態**: `game.encode(name)` が返す `scalars`／`tokens`／`card_idx` を、記録の `sc`／`tok`／`ci`
  と**同じ形**（`record_gen._Recorder.__call__` と同じ reshape・同じ `MAX_CI` パディング）に組む
  （`raw_row`）。
* **候補**: `game.decide(name, opts)` が返す `out`（`stats.legal`／`groups`／`sig`）と
  `game.dump_index_json(name)` から、**記録の `pol_sig`／`pol_cid`／`pol_tcid`／`pol_si`／`pol_ti`／
  `pol_k` と同じ規約**（`n_rel.cand_ids`・`record_gen._don_k`）で候補記述子を組む（`raw_candidates`）。
  **`record_gen._Recorder.__call__` の候補構築部と 1 行ずつ同じ**（独立実装ではなく意図的な再利用——
  ここがずれていたら「候補ごとの理論値」がそもそも記録と紐付かない）。
* **理論値**: 状態＋候補記述子から、**既存の価格式をそのまま呼ぶ**——
  `kappa_vector.state_of_row`（`Θ_me`／`Θ_opp`／`A_me`／`A_opp`。`A` は `two_curves_state.state_by_turn`
  と同じ「両席の直近の自席ターン開始行」の運び方をその場で行う）・`two_curves._price_row`
  （`theory_order.score_candidate` を帳簿の規約で読み直したもの・T137a の再利用）。

## 検算（「生 = 記録」）

**同じ seed の対局を、記録を作ったのと同じ設定（sims・net・dirichlet_eps・temp_turns・worlds・
decks）で今ここに打ち直す**（`opcg_sim.loop.driver.run_game` を直接呼ぶ・`record_gen.play_one` が
呼ぶのと同じ経路・対局は seed から決定的）。打ち直しながら `replay_and_compare` が 3 つを比べる:

1. **状態の一致**（`compare_state`）: 生の `(sc, tok, ci)` を `record_gen` と同じ float16／int16 へ
   落とし、**既存のダンプの同じ (seed, step) の行とビット一致するか**。
2. **理論値の一致**: 生の状態から読んだ `Θ_me`／`Θ_opp`／`A_me`／`A_opp`（`_LiveState.on_turn_start`）
   と、**ダンプの行から** `two_curves_state.state_by_turn` が読んだ同じ量が一致するか。
3. **候補ごとの理論値の一致**: その行の全候補について、**生の状態**（全精度・float16 へ落とす前）
   から読んだ `score_candidate` と、**ダンプの `pol_*` 列**（float16 に一度落ちている）から読んだ
   同じ候補の値が一致するか。**併せて、生の状態にダンプと同じキャストを掛けてから**同じ比較を
   もう一度行う（`max_abs_after_cast`）——全精度どうしでズレたら**配線の誤り**、キャストを
   揃えると消えるズレなら**格納精度（float16）が価格式の分岐（しきい値の判定など）を
   またいだだけ**という切り分けになる。

**予告**: 1・2 は完全に一致するはず（生成時に一度呼んだのと同じ関数を同じ引数でもう一度呼ぶだけ）。
3 は**キャストを揃えれば**一致するはずだが、**全精度どうしでは一致しないことがありうる**——
`score_candidate` は手札の計画価値など**しきい値をまたぐ判定**を含むので、float16 の丸め幅
（相対 1e-3 程度）が判定の向きを変える候補が少数出ても配線の誤りではない（予告として書いておく）。

使い方（`--in` は検算対象の記録の入ったディレクトリ・`--seed` はその記録に含まれる 1 局の seed）:

    python tests/scripts/live_theory.py --in <records_dir> --seed 8800 [--decks user] [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
import guard_afford as GA  # noqa: E402
import kappa_vector as KV  # noqa: E402
import two_curves as TC  # noqa: E402
from crossing_bridge import own_turn_index  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: 記録と同じ形（`record_gen.py` の規約——`MAX_CI`／`TOKENS_SHAPE`／`DT_V3` はそこから借りる）。
TOKENS_SHAPE = RG.TOKENS_SHAPE
DT_V3 = RG.DT_V3


def raw_row(game, name):
    """`game.encode(name)` → 記録と同じ形の `(sc, tok, ci)`（float32・キャストする前）。"""
    enc = json.loads(game.encode(name))
    sc = np.asarray(enc["scalars"], np.float32)
    tok = np.asarray(enc["tokens"], np.float32).reshape(TOKENS_SHAPE)
    ci = np.zeros(RG.MAX_CI, np.int64)
    src = np.asarray(enc["card_idx"], np.int64)[:RG.MAX_CI]
    ci[:len(src)] = src
    return sc, tok, ci


def cast_row(sc, tok, ci):
    """`record_gen` の dump v3 キャスト（float32/int64 → float16/int16）を同じ順でかける。"""
    return (sc.astype(DT_V3["scalars"]), tok.astype(DT_V3["tokens"]), ci.astype(DT_V3["card_idx"]))


def raw_candidates(game, name, out):
    """`out`（`game.decide` の戻り）の `groups` → 記録の `pol_*` と同じ規約の候補記述子の列。

    **`record_gen._Recorder.__call__` の候補構築部と同じ関数を呼ぶ**（`RG.move_sig`・`RG._don_k`・
    `NL.cand_ids`・`dump_index_json` の `cids`／`slots`）——独立実装ではなく**意図的な共有**
    （ここがずれていたら記録の `pol_*` と紐付かない）。`kind != "main"` や候補が無ければ空。"""
    groups = out.get("groups") or []
    if out.get("kind") != "main" or not groups:
        return []
    legal = (out.get("stats") or {}).get("legal") or []
    idx = json.loads(game.dump_index_json(name))
    cids, slots = idx.get("cids") or {}, idx.get("slots") or {}
    src_uuid = None
    if any((legal[g["rep"]] or {}).get("action_type") == NL.SELECT_AT for g in groups):
        src_uuid = (json.loads(game.pending_json()) or {}).get("source_card_uuid")
    cands = []
    for g in groups:
        rep = legal[g["rep"]]
        sig = RG.move_sig(rep)
        su, tu = NL.cand_ids(rep, src_uuid)
        gk = RG._don_k(rep)
        cands.append({"sig": sig, "cid": cids.get(su) or None, "tcid": (cids.get(tu) or None) if tu else None,
                     "si": slots.get(su, -1) if su else -1, "ti": slots.get(tu, -1) if tu else -1,
                     "k": -1 if gk is None else int(gk), "n": float(g["n"]), "q": float(g["q"])})
    return cands


def price_candidates(sc, tok, ci, cards, idx2cid, cands, theta=THETA, mu=MU):
    """候補ごとの理論値（`two_curves._price_row` の再利用——T137a の帳簿の規約のまま）。
    値付けできない候補は `None`（`TURN_END` 等）。"""
    out = []
    for c in cands:
        v = TC._price_row(sc, tok, ci, cards, idx2cid, c["sig"], c["cid"], c["tcid"],
                          c["si"], c["ti"], c["k"], theta, mu)
        out.append(dict(c, price=v))
    return out


class _LiveState:
    """自席ターン開始行の `A`／`g` を積みながら歩く（`two_curves_state.state_by_turn` と同じ運び方を
    **その場で**行う——過去のターンを覚えておけば次のターンで両席ぶん読める）。"""

    def __init__(self, cards, idx2cid, decks, theta=THETA, mu=MU):
        self.cards, self.idx2cid, self.decks = cards, idx2cid, decks
        self.theta, self.mu = theta, mu
        self.rate_at_turn, self.g_at_turn = {}, {}

    def on_turn_start(self, w, t, sc, tok, ci):
        """その自席ターンの最初の行で呼ぶ——`A`／`g` を記録し、両席ぶん読めれば `state_of_row` を返す
        （読めなければ `None`＝先手の最初のターンと同じ扱い）。"""
        dk = self.decks[w]
        j = own_turn_index(t)
        self.rate_at_turn[(w, t)] = KV.rate_of_row(sc, tok, ci, self.idx2cid, self.cards,
                                                   self.theta, self.mu, deck_ids=dk, j=j)
        self.g_at_turn[(w, t)] = KV.g_of_row(sc, tok, ci, self.idx2cid, self.cards)
        ts = [tt for (ww, tt) in self.rate_at_turn if ww == 1 - w and tt < t]
        if not ts:
            return None
        key = (1 - w, max(ts))
        a_opp, g_opp = self.rate_at_turn[key], self.g_at_turn[key]
        th_me, th_opp, a_me2, a_opp2, _j = KV.state_of_row(
            sc, tok, self.rate_at_turn[(w, t)], a_opp, j, g_me=self.g_at_turn[(w, t)], g_opp=g_opp)
        return {"th_me": th_me, "th_opp": th_opp, "a_me": a_me2, "a_opp": a_opp2}


def diff_candidate_prices(priced, priced_cast, off, tol=1e-4):
    """候補ごとの `(全精度の差, キャストを揃えた差, 全精度で `tol` を超えた数, 型ごとの内訳)`。

    `priced`／`priced_cast`／`off` は同じ順・同じ長さの候補の値付け列（`price_candidates` の
    戻り値または `{"price": …}` の列）——**両方とも `None` なら差は 0・片方だけ `None` なら `inf`**
    （値付けできる／できないの食い違いはそれ自体が異常値として拾う）。"""
    devs, devs_cast, n_diff_full, fam_diffs = [], [], 0, {}
    for a, ac, b in zip(priced, priced_cast, off):
        if a["price"] is None or b["price"] is None:
            devs.append(0.0 if a["price"] == b["price"] else float("inf"))
            devs_cast.append(0.0 if ac["price"] == b["price"] else float("inf"))
            continue
        d_full = abs(float(a["price"]) - float(b["price"]))
        devs.append(d_full)
        devs_cast.append(abs(float(ac["price"]) - float(b["price"])))
        if d_full > tol:
            n_diff_full += 1
            fam = (a["sig"][0] if a["sig"] else None) or "?"
            fam_diffs[fam] = fam_diffs.get(fam, 0) + 1
    return devs, devs_cast, n_diff_full, fam_diffs


def _abs_max(pairs):
    if not pairs:
        return None
    return round(max(abs(float(a) - float(b)) for a, b in pairs), 6)


def compare_state(live_cast, rec_sc, rec_tok, rec_ci):
    """生の状態（キャスト後）と記録の行のビット差（0 が期待値）。"""
    sc, tok, ci = live_cast
    return {"sc_max_abs": _abs_max([(a, b) for a, b in zip(np.asarray(sc, np.float32).ravel(),
                                                          np.asarray(rec_sc, np.float32).ravel())]),
           "tok_max_abs": _abs_max([(a, b) for a, b in zip(np.asarray(tok, np.float32).ravel(),
                                                            np.asarray(rec_tok, np.float32).ravel())]),
           "ci_n_bad": int(np.sum(np.asarray(ci, np.int64) != np.asarray(rec_ci, np.int64)))}


def replay_and_compare(dirs, seed, decks_mode, sims=64, net=None, dirichlet_eps=0.25,
                       temp_turns=4, worlds=4, theta=THETA, mu=MU):
    """記録と同じ設定で seed を打ち直し、生の読みを記録の同じ行と突き合わせる（§検算）。

    `dirs` の中に `seed` を含む記録が無ければ `ValueError`。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    rec = _load_record_rows(dirs, seed)
    if rec is None:
        raise ValueError("記録に seed=%d が無い（%s）" % (seed, dirs))

    E.engine()
    search = {"worlds": int(worlds)} if worlds else {}
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps, temp_turns=temp_turns,
                      prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)
    decks = {0: list(p1[1]), 1: list(p2[1])}
    live_st = _LiveState(cards, idx2cid, decks, theta, mu)

    n = 0
    state_devs, theta_devs, cand_devs = [], [], []
    fam_diffs = {}       # sig[0] → 全精度で `1e-4` を超えて食い違った候補の数（§検算 3・下の注記）
    counts = {"rows": 0, "rows_out_of_range": 0, "rows_kind_mismatch": 0, "state_rows_matched": 0,
             "theta_rows_matched": 0, "theta_rows_no_pair": 0, "theta_rows_missing_offline": 0,
             "cand_rows_matched": 0, "cand_rows_len_mismatch": 0, "cand_n": 0,
             "cand_n_diff_full": 0, "cand_n_diff_after_cast": 0}

    def observer(game, name, turn, step, out, move):
        nonlocal n
        w = 0 if name == "p1" else 1
        sc, tok, ci = raw_row(game, name)
        cast = cast_row(sc, tok, ci)
        counts["rows"] += 1
        kind = RG._KIND.get(out.get("kind"), 0)
        if n >= len(rec["who"]):
            counts["rows_out_of_range"] += 1
            n += 1
            return
        # **同じ手順で歩いているはずなので、`n` 番目の記録行と (who, turn, kind) が揃うはず**
        # （揃わなければ配線の誤り——`rows_kind_mismatch` に数えて次に進む。状態の突き合わせは
        # 諦めない：ずれてもこの行自体は記録の `n` 番目と比べる）。
        if (int(rec["who"][n]), int(rec["turn"][n]), int(rec["kind"][n])) != (w, int(turn), kind):
            counts["rows_kind_mismatch"] += 1
        d = compare_state(cast, rec["sc"][n], rec["tok"][n], rec["ci"][n])
        state_devs.append(d)
        if (d["sc_max_abs"] or 0.0) < 1e-9 and (d["tok_max_abs"] or 0.0) < 1e-9 and d["ci_n_bad"] == 0:
            counts["state_rows_matched"] += 1
        if out.get("kind") == "main":
            first_of_turn = PL.is_own_turn(w, int(turn)) and (w, int(turn)) not in live_st.rate_at_turn
            if first_of_turn:
                live_theta = live_st.on_turn_start(w, int(turn), sc, tok, ci)
                off_theta = rec["theta"].get(n)
                if live_theta is not None and off_theta is not None:
                    dd = {k: round(abs(live_theta[k] - off_theta[k]), 6) for k in live_theta}
                    theta_devs.append(dd)
                    if max(dd.values()) < 1e-6:
                        counts["theta_rows_matched"] += 1
                elif live_theta is None and off_theta is None:
                    counts["theta_rows_no_pair"] += 1
                else:
                    counts["theta_rows_missing_offline"] += 1
            cands = raw_candidates(game, name, out)
            if cands:
                # **全精度**（`raw_row` の float32・キャストしていない）——T141／T18 が実際に使う形。
                priced = price_candidates(sc, tok, ci, cards, idx2cid, cands, theta, mu)
                # **ダンプと同じキャストを掛けてから**（float16 → float32 に戻す・記録と同じ丸め）。
                # **全精度との差が「配線の誤り」か「格納精度の丸め」かを切り分けるための対照**。
                c_sc, c_tok, c_ci = cast_row(sc, tok, ci)
                priced_cast = price_candidates(c_sc.astype(np.float32), c_tok.astype(np.float32),
                                              c_ci.astype(np.int64), cards, idx2cid, cands, theta, mu)
                off = rec["cands"].get(n) or []
                if len(off) != len(priced):
                    counts["cand_rows_len_mismatch"] += 1
                else:
                    devs, devs_cast, n_diff_full, row_fam_diffs = diff_candidate_prices(priced, priced_cast, off)
                    counts["cand_n_diff_full"] += n_diff_full
                    for fam, cnt in row_fam_diffs.items():
                        fam_diffs[fam] = fam_diffs.get(fam, 0) + cnt
                    counts["cand_n_diff_after_cast"] += sum(1 for d in devs_cast if d > 1e-4)
                    cand_devs.append({"n": len(devs), "max_abs": round(max(devs), 6) if devs else None,
                                     "max_abs_after_cast": round(max(devs_cast), 6) if devs_cast else None})
                    counts["cand_n"] += len(devs)
                    if devs and max(devs) < 1e-4:
                        counts["cand_rows_matched"] += 1
        n += 1

    res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=observer)
    counts["winner"] = res["winner"]
    counts["turns"] = res["turns"]
    counts["rows_recorded"] = len(rec["who"])
    return {
        "counts": counts,
        "state": {"n": len(state_devs),
                 "sc_max_abs": max((d["sc_max_abs"] or 0.0) for d in state_devs) if state_devs else None,
                 "tok_max_abs": max((d["tok_max_abs"] or 0.0) for d in state_devs) if state_devs else None,
                 "ci_n_bad_total": sum(d["ci_n_bad"] for d in state_devs)},
        "theta": {"n": len(theta_devs),
                 "max_abs": (max(max(d.values()) for d in theta_devs) if theta_devs else None)},
        "candidates": {"n_rows": len(cand_devs), "n_candidates": counts["cand_n"],
                      # **全精度**（`raw_row` のまま）対 ダンプ——格納の丸めを含めた差
                      "max_abs": (max((d["max_abs"] or 0.0) for d in cand_devs) if cand_devs else None),
                      "n_diff_full": counts["cand_n_diff_full"],
                      # **同じキャストを掛けてから**（配線だけの検算・§検算の主）
                      "max_abs_after_cast": (max((d["max_abs_after_cast"] or 0.0) for d in cand_devs)
                                            if cand_devs else None),
                      "n_diff_after_cast": counts["cand_n_diff_after_cast"],
                      "diff_full_by_family": dict(sorted(fam_diffs.items(), key=lambda kv: -kv[1]))},
    }


def _load_record_rows(dirs, seed):
    """`dirs` から `seed` の main 行だけを step 順に読み、`(sc,tok,ci)`・`theta` 状態・候補の値付けを
    オフライン（ダンプ経由）で作る（`replay_and_compare` の突き合わせの相手側）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        if seed_g != seed:
            continue
        order = list(idx)
        out = {"who": [], "turn": [], "kind": [], "sc": [], "tok": [], "ci": [], "theta": {}, "cands": {}}
        seat_decks = KV._seat_decks(dirs)
        st_by_turn = TS_state_by_turn_cache(rows, ex, idx, cards, idx2cid, seat_decks, seed_g)
        for n, i in enumerate(order):
            w, t, kind = int(rows["who"][i]), int(rows["turn"][i]), int(rows["kind"][i])
            out["who"].append(w); out["turn"].append(t); out["kind"].append(kind)
            out["sc"].append(ex["sc"][i]); out["tok"].append(ex["tok"][i]); out["ci"].append(ex["ci"][i])
            if kind == 0:
                st = st_by_turn.get((w, t))
                if st is not None:
                    out["theta"][n] = {"th_me": st["th_me"], "th_opp": st["th_opp"],
                                       "a_me": st["a_me"], "a_opp": st["a_opp"]}
                k = int(L[i])
                if k > 0:
                    b = int(ptr[i])
                    cands = []
                    for j in range(b, b + k):
                        sig = json.loads(pol["pol_sig"][j])
                        tl = sig[2] if len(sig) > 2 else None
                        cid, tcid = str(pol["pol_cid"][j]) or None, (str(pol["pol_tcid"][j]) or None) if tl else None
                        price = TC._price_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], cards, idx2cid,
                                              sig, cid, tcid, int(pol["pol_si"][j]), int(pol["pol_ti"][j]),
                                              int(pol["pol_k"][j]), THETA, MU)
                        cands.append({"price": price})
                    out["cands"][n] = cands
        return out
    return None


def TS_state_by_turn_cache(rows, ex, idx, cards, idx2cid, seat_decks, seed_g):
    """`two_curves_state.state_by_turn` を呼ぶだけの薄い包み（循環 import を避けるため遅延 import）。"""
    import two_curves_state as TS
    return TS.state_by_turn(rows, ex, idx, cards, idx2cid, seat_decks, seed_g)


def build_parser():
    ap = argparse.ArgumentParser(description="理論を生の局面から読む橋と「生=記録」の検算（T140）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True, help="打ち直す対局の seed（記録に含まれるもの）")
    ap.add_argument("--decks", default=None, help="省略時は meta_n_record.json の decks を使う")
    ap.add_argument("--json", default="")
    return ap


def _decks_mode_of(dirs, given):
    if given:
        return given
    for d in dirs:
        p = os.path.join(d, "meta_n_record.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("decks") or "synth"
    return "synth"


def main(argv=None):
    a = build_parser().parse_args(argv)
    decks_mode = _decks_mode_of(a.src, a.decks)
    out = replay_and_compare(a.src, a.seed, decks_mode)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
