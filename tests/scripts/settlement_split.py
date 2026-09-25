#!/usr/bin/env python3
"""**T139**（2026-09-23）: 「詰みを逃した」（CPU の見落とし・打ち筋・§8.5）か
「定義に無い守り・打点がある」（理論の穴）かを記録から機械的に切り分ける。

## 問い

T136（`lethal_rule.py`・決着の定義）が両記録で残した 2 種類の不一致:

* **敗者側の宣言**（`declared=True` で宣言した席が実際には**負けた**）——実 13 行／合成 55 行
  （`docs/reports/2026-09-23_settlement_definition.md` §0-5）。
* **取りこぼし**（勝者の**実際の最終ターン**なのに `declared=False`）——実 38 行／合成 177 行（同 §0-4）。

**`precision_winner` が 1 に届かない分**（T136 の器の docstring）は (a) 定義の誤りか (b) 詰みが在ったのに
CPU が決めなかったかのどちらかだが、**記録からは分けられない**、と書いていた。本 T はその切り分けを実装する。

## 式（新定数ゼロ・規則だけ）

そのターンに**実際に選んだ**攻撃の本数 `attacks_made` を数える——候補の署名（`pol_sig`）が
`theory_bridge.move_family` で `"attack"`（`ATTACK` または対象付き `DON_BOX`）と分類される選択の数
（`theory_bridge.py` の T86 `atk_turn_n` と**同じ数え方**）。これを、ターン開始時点で攻撃できた体の数
`xs`（`lethal_rule.lethal_of_row` が返す内訳の `xs`＝`own_attackers_of` の長さ）と比べる:

* **`attacks_made >= xs`（全力で殴った）** … 攻め手は取れる手を尽くした。それでも理論の計算（通る本数・
  止められる本数）と実際の結果が食い違った＝**定義の穴の候補**（攻撃時の能力・トリガー・保護など、
  `lethal_of_row` が数えていない機構が実在する可能性）。
* **`attacks_made < xs`（全力ではなかった）** … 攻め手は攻撃できる体を残した。
  * **敗者側の宣言**でこれが起きた場合 … CPU が「詰み」の候補列を選ばなかった＝**CPU の見落とし候補**
    （§8.5・打ち筋の課題。ただし理論の宣言自体が正しかったかは別途要検証——候補に「その全力の一手」が
    実在したかは本器では確認しない）。
  * **取りこぼし**でこれが起きた場合 … 全力でなくても実際には勝てた＝**定義が守りを過大に見込んでいた
    候補**（T136 §0.4 の「87〜88% は守り手に止める本数を与えている」と符合するはずの行）。

## 測るもの

敗者側の宣言／取りこぼしの 2 カテゴリ × `full_swing` の有無で 4 セル（両記録）。
`xs` と `attacks_made` の差の分布・代表行のサンプルも出す。

使い方:

    python tests/scripts/settlement_split.py --in <records_dir> [--games N] [--json out.json]
"""

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import lethal_rule as LR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, move_family  # noqa: E402


def attacks_made_by_turn(dirs, limit_games=0):
    """局×席×ターン → **実際に選んだ**攻撃の本数（`move_family(選んだ候補) == "attack"` の行数）。
    **`theory_bridge` の T86 `atk_turn_n` と同じ数え方**（`kind==0`・自席ターン・候補が選ばれている行）。
    `k>=1`（1 候補しか無い強制手も数える——`theory_bridge` の `k>=2`〔スコアリング用の制約〕は要らない）。"""
    out = {}
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(r["seed"][idx[0]]) if len(idx) else -1
        for i in idx:
            w, t = int(r["who"][i]), int(r["turn"][i])
            if int(r["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            k = int(L[i]); ch = int(r["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sig = json.loads(pol["pol_sig"][int(ptr[i]) + ch])
            if move_family(sig) == "attack":
                key = (seed_g, w, t)
                out[key] = out.get(key, 0) + 1
    return out


def _classify(rows, attacks_made):
    """`rows`（lethal_rule の dump の行の部分集合）を `full_swing` で 2 分し、集計と代表行を返す。

    **`margin`**（`hits − (life + 1)`）も添える——**事後の観察**（測る前の予告には無かった）: 宣言の際どさ
    （余裕 0＝ぎりぎり）が誤判定に効くかを見るための診断で、式の一部ではない（`declared` の判定は変えない）。"""
    full, partial = [], []
    for r in rows:
        n_atk = attacks_made.get((r["seed"], r["w"], r["t"]), 0)
        rec = {"seed": r["seed"], "w": r["w"], "t": r["t"], "t_end": r["t_end"],
               "life": r["life"], "xs": r["xs"], "attacks_made": n_atk,
               "through": r["through"], "stops": r["stops"], "hits": r["hits"],
               "margin": int(r["hits"] - (r["life"] + 1))}
        (full if n_atk >= r["xs"] else partial).append(rec)
    margin_hist = {}
    for r in full + partial:
        k = str(r["margin"])
        margin_hist[k] = margin_hist.get(k, 0) + 1
    return {"n": len(rows), "full_swing": len(full), "partial_swing": len(partial),
            "full_swing_share": round(len(full) / max(1, len(rows)), 4),
            "margin_hist": margin_hist,
            "full_swing_sample": full[:10], "partial_swing_sample": partial[:10]}


def collect(dirs, limit_games=0, with_don=True):
    dump = []
    lr_out = LR.collect(dirs, limit_games, with_don, dump=dump)
    attacks_made = attacks_made_by_turn(dirs, limit_games)
    false_declared = [r for r in dump
                      if r["declared"] and r["winner"] is not None and r["w"] != r["winner"]]
    missed = [r for r in dump
             if (not r["declared"]) and r["winner"] == r["w"] and r["t"] == r["t_end"]]
    return {"lethal_rule": {"declared": lr_out["declared"], "precision_winner": lr_out["precision_winner"],
                            "false_declared": lr_out["false_declared"]},
            "false_declared": _classify(false_declared, attacks_made),
            "missed_lethal": _classify(missed, attacks_made)}


def build_parser():
    ap = argparse.ArgumentParser(description="詰みの見落としと定義の穴を切り分ける（T139）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--don", default="on", choices=("on", "off"))
    ap.add_argument("--hand", default=None, choices=LR.LETHAL_HAND_MODES,
                    help="lethal_rule の --hand をそのまま通す（既定 actual）")
    ap.add_argument("--stop", default=None, choices=LR.LETHAL_STOP_MODES,
                    help="lethal_rule の --stop をそのまま通す（既定 max・missed_lethal の "
                         "margin=-1 が `econ`〔T130 の受けるより安いときだけ切る〕でどう動くかの検算に使う）")
    ap.add_argument("--life", default=None, choices=LR.LETHAL_LIFE_MODES,
                    help="lethal_rule の --life をそのまま通す（既定 draw）")
    ap.add_argument("--avg-counter", default=None, choices=LR.AVG_COUNTER_MODES,
                    help="lethal_rule の --avg-counter をそのまま通す（既定 printed）")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.hand:
        LR.set_lethal_hand_mode(a.hand)
    if a.stop:
        LR.set_lethal_stop_mode(a.stop)
    if a.life:
        LR.set_lethal_life_mode(a.life)
    if a.avg_counter:
        LR.set_avg_counter_mode(a.avg_counter)
    out = collect(a.src, a.games, a.don == "on")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
