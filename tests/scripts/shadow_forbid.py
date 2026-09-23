#!/usr/bin/env python3
"""**T141**（2026-09-23）: **影の介入（shadow）**——打ち回しは変えずに「理論なら禁じた手」の
率と型を数える。

## 問い

T18（出口）は「理論が純損と言う手を禁じて勝率が動くか」を測る（規則・完全情報の理論を打ち回しに
**介入させる**）。介入する前に、**介入しなければ何が起きているか**を先に知っておく必要がある——
理論が「純損」と呼ぶ手は実際どれくらいの頻度で、どんな型で起きているか。本 T はそれを
**打ち回しには一切介入せず**（`swap` を使わない・観測のみ）、T140 で作った「理論を生の局面から
読む橋」の上で測る。**T140 で見つかった `PLAY` の全精度の不安定さもここで踏まえる**——キャストを
揃えた値との差も併せて出す（`shadow_row` の `s_cast`／`forbidden_cast`）。

## 予告（測る前に書く・2 局の下見で見えた形を踏まえる）

**式を固める前に 2 局だけ試し打ちし、`attach`〔`DON_BOX` の純付与〕の禁じ率が極端に高いことに
気づいた**。理由は構造的——`score_candidate` は**その 1 手だけの静的な価格**しか見ない
（T137a の docstring どおり）。だが「まずドンを付けて、次のターン以降にそのドンで殴る」計画は
**2 手にまたがる**——1 手目（付与）の静的な価格は、2 手目（付与ぶん強くなった攻撃）の価格より
低く出やすい。search は先読みで系列の価値を見るが `score_candidate` は見ない。**この非対称は
「CPU が悪い」でも「理論が間違っている」でもなく、`s` という物差しの守備範囲の外**（系列の価値は
測っていない）。この下見を踏まえて予告する:

1. **全体の禁じ率は無視できない大きさになる**（下見の 2 局で 55%）。
2. **`attach` の禁じ率が突出して高い**（下見で 100%）——上の機構どおりなら、**`attach` を除くと
   禁じ率は大きく下がる**。
3. **禁じられた行で理論が薦める型は `attack` が最多**（`attack` の価格が最も高く出やすい・
   T81／T83 で最も較正の効いた型でもある）。
4. **キャストを揃えても判定はほとんど変わらない**（T140 の「食い違いは 0.2%」がここでも効く）。

## 式（新定数ゼロ・既存の量の再利用のみ）

`s = played_v − max(scored)`（`theory_bridge.collect` が記録から積んでいる `s_row` と**同じ定義**——
`played_v` は実際に選んだ候補の理論値・`max(scored)` はその行で値付けできた候補の最大値。定義上
`s ≤ 0`。**`s < 0`（ごく小さい許容 `tol` を超えて）を「理論が禁じる」と呼ぶ**——他に厳密に上回る
候補が在ったという意味で、新しい判断基準ではなく既存の `s_row` の符号をそのまま読むだけ。

**型**は `theory_bridge.move_family`（T81 の既存の分類）——実際に選んだ手の型 `played_family` と、
理論が薦めた手の型 `best_family`（`scored` の argmax）を両方記録する。**候補が 2 本未満・値付け
できない行は母数から除く**（`theory_bridge.collect` と同じ規約——無言の行を混ぜない）。

## 生成（記録の再生ではなく、新しい seed で今ここに打つ）

T140 は既存の記録の seed を打ち直して「生=記録」を検算した。本 T は**まだ一度も記録されていない
新しい seed**で対局を生成しながら測る——T18／T142 は arena で新しく打った対局を使うので、
「まだ記録が無い局面でも測れる」ことを本 T で確かめておく（既存記録の再生に頼らない経路）。

## T144（2026-09-23）: 付与を 2 手の系列として読む（`--seq attack`）

T141 で `attach` の禁じ率は 93%／91% だった。原因は単位のずれ——付与の静的価格は「攻撃の価値の
**増分**」、攻撃候補は「**総額**」で、同じ行に並べると付与は構造的に負ける。規則上、付与したドンは
自分のターン中だけ効き、価値は同じターンにその k 枚を乗せて殴る攻撃にしかない。その攻撃は同じ行に
`DON_BOX(同じ uuid, 対象, 同じ k)` として在る。`seq_prices` は純付与の価格をその攻撃候補の最大価格に
読み替える（式は関数の docstring・新定数ゼロ）。

**予告（測る前に書く）**:

1. **`attach` の禁じ率は大きく下がり、`attack` の禁じ率（T141 で 49%／40%）の近くまで落ちる**——
   読み替えた付与は、自分が準備する攻撃と同じ価格になるので、その攻撃より上の候補が在るときだけ禁じられる。
2. **`attach` 以外の型の禁じ率は下がらない**（読み替えは純付与の価格を上げるだけ＝行の最大値は
   上がるか据え置き。付与が最大値になる行では他の型の `s` が下がり、禁じ率はむしろ上がりうる）。
3. **読み替えできない純付与（同じ uuid・同じ k の攻撃候補が無い）が一定数残る**——出たら件数を書く。

使い方:

    python tests/scripts/shadow_forbid.py --games 10 --seed-base 50000 --decks user [--json out.json]
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

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
import guard_afford as GA  # noqa: E402
import live_theory as LT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from theory_bridge import move_family  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: `s < -TOL` を「理論が禁じる」と呼ぶ（浮動小数の丸め誤差を吸収するだけ・新しい判断基準ではない）。
TOL = 1e-9


#: **T144**（2026-09-23）: 純付与を 2 手の系列として読むか。`off`＝T141 のまま（静的な `attach_value`）／
#: `attack`＝**同じカード・同じ枚数の攻撃候補の最大価格**に読み替える（下の `seq_prices`）／
#: `attack_le`＝**同じカード・枚数が同じか少ない攻撃候補の最大価格**（測った後に足した第 2 の読み・T144 §4）。
SEQ_MODES = ("off", "attack", "attack_le")
SEQ_MODE = "off"


def set_seq_mode(mode):
    global SEQ_MODE
    if mode not in SEQ_MODES:
        raise ValueError("SEQ_MODE は %r のどれか（%r）" % (SEQ_MODES, mode))
    SEQ_MODE = mode


def _is_pure_attach(sig):
    return bool(sig) and sig[0] in ("DON_BOX", "ATTACH_DON") and not (len(sig) > 2 and sig[2])


def _is_box_attack(sig):
    return bool(sig) and sig[0] == "DON_BOX" and len(sig) > 2 and bool(sig[2])


def _k_of(c, default):
    return int(c["k"]) if c["k"] is not None and c["k"] >= 0 else default


def seq_prices(cands, priced, mode=None):
    """**純付与の価格を、それが準備する攻撃の価格に読み替える**（T144・新定数ゼロ）。

    規則: 付与したドンは**自分のターン中だけ** +1000 で、次のリフレッシュで戻る＝純付与そのものは
    損害を生まない。価値は「同じターンにそのカードがその k 枚を乗せて殴る攻撃」にしかない。
    その攻撃は同じ行の候補に `DON_BOX(同じ uuid, 対象, 同じ k)` として既に在り、
    `score_candidate` が `+1000k` 込みで値付けしている。静的な `attach_value` は**攻撃の価値の増分**
    （`theory_order.attach_value`）で、攻撃候補は**総額**——**同じ行に並べると単位がずれ、付与は
    構造的に必ず負ける**（T141 の `attach` の禁じ率 93% の正体）。

    読み替え: 純付与（uuid=X・k 枚）の価格 ← `max{price(c) : c は DON_BOX 攻撃・c.uuid=X・c.k=k}`。
    同じ X・同じ k の攻撃候補が無ければ**静的価格のまま**（今のターンに殴れない＝【ドン!!×N】の
    条件付け等の別の用途・T74 が扱う）。`ATTACH_DON`（1 枚）は k=1 として扱う。

    **`attack_le`**（測った後の第 2 の読み）: 箱の生成器（`rust/.../search/macro.rs`）は攻撃の箱に
    **k ∈ {0, 相手を越える最小, カウンター 2 枚要求}**、純付与の箱に **k ∈ {1, 全部, 【ドン!!×N】の不足分}** しか
    出さない＝2 つの組はめったに重ならない（1 枚の付与に 1 枚の攻撃候補が在るのは、1 枚でちょうど越えるときだけ）。その付与の
    攻撃としての価値は「しきい値を越えない分は変わらない」＝**枚数が同じか少ない攻撃候補の最大価格**で読む。
    `mode` を省けば `SEQ_MODE` に従う。戻り値: `(読み替えた priced, 読み替えた件数, 純付与の件数)`。"""
    mode = SEQ_MODE if mode is None else mode
    atk = {}
    for c, p in zip(cands, priced):
        if p["price"] is None or not _is_box_attack(c["sig"]):
            continue
        atk.setdefault(c["sig"][1], []).append((_k_of(c, 0), float(p["price"])))
    out, n_re, n_attach = [], 0, 0
    for c, p in zip(cands, priced):
        if _is_pure_attach(c["sig"]) and p["price"] is not None:
            n_attach += 1
            k = _k_of(c, 1)
            vs = [pr for kk, pr in atk.get(c["sig"][1], [])
                  if (kk <= k if mode == "attack_le" else kk == k)]
            v = max(vs) if vs else None
            if v is not None:
                out.append(dict(p, price=v))
                n_re += 1
                continue
        out.append(p)
    return out, n_re, n_attach


def _reread_index(cands, i, priced, mode=None):
    """候補 `i`（純付与）が読み替えの対象になった（同じ uuid・読みに合う k の攻撃候補が在った）か。"""
    mode = SEQ_MODE if mode is None else mode
    c = cands[i]
    k = _k_of(c, 1)
    return any(_is_box_attack(o["sig"]) and o["sig"][1] == c["sig"][1]
               and (_k_of(o, 0) <= k if mode == "attack_le" else _k_of(o, 0) == k)
               and p["price"] is not None for o, p in zip(cands, priced))


def find_chosen(cands, out, move):
    """`out`／`move` から実際に選んだ候補の index（無ければ `None`）。

    **`record_gen._Recorder.__call__` の照合と同じ**——box レベルの `sig`（`out["sig"]` 優先・
    無ければ `RG.move_sig(move)`）と、生の `don_k`（`k_raw`・**`-1` 化する前**）の組で照合する。"""
    sig = out.get("sig") if out.get("sig") is not None else RG.move_sig(move)
    k_sel = out.get("k")
    for i, c in enumerate(cands):
        if c["sig"] == sig and c.get("k_raw") == k_sel:
            return i
    return None


def _best(priced_list):
    scored = [(i, p["price"]) for i, p in enumerate(priced_list) if p["price"] is not None]
    if len(scored) < 2:
        return None
    return scored


def shadow_row(sc, tok, ci, cards, idx2cid, cands, out, move, theta=THETA, mu=MU):
    """1 main 行ぶんの影の判定。値付けできない・候補 2 本未満・選んだ候補が見つからなければ
    `None`（母数から除く——`theory_bridge.collect` の無言行の扱いと同じ）。

    **T140 の所見**（`PLAY` は格納精度〔float16〕で価格が判定の分岐をまたぐことがある）を踏まえ、
    **全精度**（`sc`／`tok`／`ci` のまま）と**記録と同じキャストを揃えた値**の両方で判定し、
    `forbidden`（全精度）と `forbidden_cast`（キャスト後）を両方返す——`s` が 0 のすぐそばの行だけ、
    キャストで判定が入れ替わりうる（新しい判断基準ではなく、T140 の対照をそのまま使う）。"""
    priced = LT.price_candidates(sc, tok, ci, cards, idx2cid, cands, theta, mu)
    n_re = 0
    if SEQ_MODE != "off":
        priced, n_re, _n_attach = seq_prices(cands, priced)
    scored = _best(priced)
    if scored is None:
        return None
    chosen = find_chosen(cands, out, move)
    if chosen is None:
        return None
    played = next((v for i, v in scored if i == chosen), None)
    if played is None:                     # 選んだ候補が値付けできなかった（`TURN_END` 等）
        return None
    best_i, best_v = max(scored, key=lambda iv: iv[1])
    s = float(played) - float(best_v)

    c_sc, c_tok, c_ci = LT.cast_row(sc, tok, ci)
    priced_cast = LT.price_candidates(c_sc.astype(np.float32), c_tok.astype(np.float32),
                                     c_ci.astype(np.int64), cards, idx2cid, cands, theta, mu)
    if SEQ_MODE != "off":
        priced_cast, _r, _a = seq_prices(cands, priced_cast)
    scored_cast = _best(priced_cast)
    if scored_cast is not None:
        played_c = next((v for i, v in scored_cast if i == chosen), None)
        _bi_c, best_v_c = max(scored_cast, key=lambda iv: iv[1])
        s_cast = (float(played_c) - float(best_v_c)) if played_c is not None else None
    else:
        s_cast = None

    return {"s": s, "forbidden": bool(s < -TOL),
           "s_cast": s_cast, "forbidden_cast": (bool(s_cast < -TOL) if s_cast is not None else None),
           "played_family": move_family(cands[chosen]["sig"]),
           "best_family": move_family(cands[best_i]["sig"]),
           # `cands` への index（`chosen`＝実際に選んだ候補・`best_index`＝理論の最善）。
           # **T142** が「置き換える」ときに `out["groups"][best_index]["rep"]` から実際の手を引くのに使う。
           "chosen_index": chosen, "best_index": best_i,
           # **T144**: 選んだ手が純付与で、攻撃の価格に読み替えられたか（`SEQ_MODE` が `off` 以外のときだけ真になりうる）
           "played_reread": bool(SEQ_MODE != "off" and n_re > 0
                                 and _is_pure_attach(cands[chosen]["sig"])
                                 and _reread_index(cands, chosen, priced)),
           "n_cands": len(scored)}


def collect(seeds, decks_mode, sims=64, net=None, dirichlet_eps=0.25, temp_turns=4, worlds=4,
           theta=THETA, mu=MU):
    """`seeds` を今ここに打ちながら影の判定を集め、局ごとの行を返す（打ち回しには介入しない）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    E.engine()
    search = {"worlds": int(worlds)} if worlds else {}
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps, temp_turns=temp_turns,
                      prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    rows = []
    n_games = n_dropped = 0
    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)

        def observer(game, name, turn, step, out, move):
            if out.get("kind") != "main":
                return
            sc, tok, ci = LT.raw_row(game, name)
            cands = LT.raw_candidates(game, name, out)
            if not cands:
                return
            row = shadow_row(sc, tok, ci, cards, idx2cid, cands, out, move, theta, mu)
            if row is not None:
                rows.append(row)

        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=observer)
        if res["winner"] is None:
            n_dropped += 1
        n_games += 1
    return rows, {"n_games": n_games, "n_dropped": n_dropped}


def summarise(rows):
    """`rows`（`shadow_row` の列）→ 全体・型別の率と大きさの表。空なら `{"n": 0}`。"""
    if not rows:
        return {"n": 0}
    n = len(rows)
    n_forbid = sum(1 for r in rows if r["forbidden"])
    s_forbid = [-r["s"] for r in rows if r["forbidden"]]         # 正の大きさで持つ
    by_played = {}
    for r in rows:
        b = by_played.setdefault(r["played_family"], {"n": 0, "n_forbid": 0})
        b["n"] += 1
        b["n_forbid"] += int(r["forbidden"])
    for b in by_played.values():
        b["rate"] = round(b["n_forbid"] / b["n"], 4) if b["n"] else None
    best_of_forbidden = {}
    for r in rows:
        if not r["forbidden"]:
            continue
        best_of_forbidden[r["best_family"]] = best_of_forbidden.get(r["best_family"], 0) + 1
    # **T140 の対照**: キャストを揃えても `forbidden` の判定が変わらない行の割合（`s_cast` が読めた行だけ）。
    cast_rows = [r for r in rows if r.get("s_cast") is not None]
    n_cast_agree = sum(1 for r in cast_rows if r["forbidden"] == r["forbidden_cast"])
    return {
        "n": n, "n_forbid": n_forbid, "rate": round(n_forbid / n, 4),
        "s_forbid_mean": round(float(np.mean(s_forbid)), 4) if s_forbid else None,
        "s_forbid_median": round(float(np.median(s_forbid)), 4) if s_forbid else None,
        "by_played_family": {k: by_played[k] for k in sorted(by_played, key=lambda k: -by_played[k]["n"])},
        "best_family_when_forbidden": dict(sorted(best_of_forbidden.items(), key=lambda kv: -kv[1])),
        "cast_agreement": {"n": len(cast_rows), "n_agree": n_cast_agree,
                          "rate": round(n_cast_agree / len(cast_rows), 4) if cast_rows else None},
    }


def build_parser():
    ap = argparse.ArgumentParser(description="影の介入——理論なら禁じた手の率と型を数える（T141）")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed-base", type=int, required=True,
                    help="既存の記録の seed 帯と重ならない値にする（新しい局を打つ）")
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth", "synth_dig",
                                                          "synth_roles", "user"))
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--seq", default="off", choices=SEQ_MODES,
                    help="T144: 純付与を攻撃の価格に読み替える（attack）か T141 のまま（off）")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    set_seq_mode(a.seq)
    seeds = [a.seed_base + i for i in range(a.games)]
    rows, meta = collect(seeds, a.decks, sims=a.sims)
    out = {"meta": meta, "summary": summarise(rows)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
