"""**P8-7(b)**: T139 が残した「全力で殴って負けた行は守り手側に定義に無い何かがある」を、
`aux_def`（P8・dump v5）で読む（2026-09-25・読み取り専用・新定数ゼロ）。

## 問い

T139（`2026-09-23_settlement_split.md`）は「敗者側の宣言」（理論が決着と宣言した席が実際には負けた行）を
`attacks_made >= xs`（全力で殴った）で 2 分し、**全力で殴った行**は「攻め手は取れる手を尽くしたのに理論の
計算（`stops`／`through`）と実際の結果が食い違った＝定義の穴の候補」と読み、**守り手側に定義に無い何かが
ある話**として P8 待ちに置いた。`lethal_rule.lethal_of_row` の `stops` は「守り手の手札のカウンター値から
計算した理論上の最大の止め方」——**実際に何回ブロック・カウンターを切ったか**は記録に無かった（`aux_tok` は
攻撃側〔相手〕の枠しか数えない）。`aux_def`（自分側の枠）でこれが読める。

## 式（新定数ゼロ・読むだけ）

攻め手の手番 `(w, t)` の**守り手**（`d = 1-w`）の**直前の自席ターン開始の行**で `aux_def`／`aux_def_row` を
読む——`aux_def` の窓は「この行の後、相手の手番が 1 回終わるまで」（`_fill_def` の規約）なので、守り手の
`t' < t`（`d` の直前の自席ターン）の行で読めば、窓はちょうど攻め手の手番 `(w, t)` に一致する。

集める量: 狙われた回数の合計・ブロックした回数の合計・失ったライフ（リーダー枠）・場を離れた体の数
（戦闘／効果）・そのターンのカウンター回数・ブロック回数（`aux_def_row`）。**理論の `stops`／`through`／`hits`
と並べて出すだけ**——単位を無理に揃えて 1 つの式にはしない（`stops` は「止めた攻撃の本数」・`counters_used`
は「切ったカウンターの枚数」で、1 本を複数枚で止めることもあるため直接比べられない・観察のための並置）。

使い方:
  OPCG_LOG_SILENT=1 python tests/scripts/t139_defender_check.py --in <n_records dump v5>... [--json out.json]
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

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.loop.record_gen import (LEFT_DEST_TRASH_BATTLE, LEFT_DEST_TRASH_EFFECT,  # noqa: E402
                                      LEFT_DEST_HAND, LEFT_DEST_DECK)
import lethal_rule as LR  # noqa: E402
import settlement_split as SS  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402

_EFFECT_DESTS = (LEFT_DEST_TRASH_EFFECT, LEFT_DEST_HAND, LEFT_DEST_DECK)


def _extra_def(dd, n):
    """`theory_bridge._extra` ＋ `aux_def`／`aux_def_row`（無い波は `None`・P8・dump v5）。"""
    out = dict(_extra(dd, n))
    out["aux_def"] = np.asarray(dd["aux_def"])[:n].astype(np.float32) if "aux_def" in dd.files else None
    out["aux_def_row"] = np.asarray(dd["aux_def_row"])[:n].astype(np.float32) if "aux_def_row" in dd.files else None
    return out


def defender_actuals_by_turn(dirs, limit_games=0):
    """局×(攻め手の視点 `(w, t)`) → 守り手が**その窓で実際にしたこと**（`aux_def` の集計）。
    `aux_def` が無い波・守り手がまだ自席ターンを打っていない（先手 1 ターン目相手）行は出さない。"""
    out = {}
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra_def):
        games += 1
        if limit_games and games > limit_games:
            break
        if ex.get("aux_def") is None:
            continue
        seed_g = int(r["seed"][idx[0]]) if len(idx) else -1
        turn_start_i = {}
        for i in idx:
            w, t = int(r["who"][i]), int(r["turn"][i])
            if int(r["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if (w, t) not in turn_start_i:
                turn_start_i[(w, t)] = i
        turn_seq = {0: sorted(t for (ww, t) in turn_start_i if ww == 0),
                   1: sorted(t for (ww, t) in turn_start_i if ww == 1)}
        for w in (0, 1):
            d = 1 - w
            for t in turn_seq[w]:
                prev = [tt for tt in turn_seq[d] if tt < t]
                if not prev:
                    continue
                i_def = turn_start_i[(d, prev[-1])]
                ad = ex["aux_def"][i_def]        # (6, 4)
                adr = ex["aux_def_row"][i_def]   # (2,)
                dests = ad[:, 3]
                out[(seed_g, w, t)] = {
                    "targeted": round(float(ad[:, 0].sum()), 3),
                    "blocked": round(float(ad[:, 1].sum()), 3),
                    "life_lost": round(float(ad[0, 2]), 3),
                    "left_battle": int(np.sum(dests == LEFT_DEST_TRASH_BATTLE)),
                    "left_effect": int(np.sum(np.isin(dests, _EFFECT_DESTS))),
                    "counters_used": round(float(adr[0]), 3),
                    "blocks": round(float(adr[1]), 3),
                }
    return out


def collect(dirs, limit_games=0, with_don=True):
    dump = []
    LR.collect(dirs, limit_games, with_don, dump=dump)
    attacks_made = SS.attacks_made_by_turn(dirs, limit_games)
    defender_actual = defender_actuals_by_turn(dirs, limit_games)
    false_declared = [r for r in dump
                      if r["declared"] and r["winner"] is not None and r["w"] != r["winner"]]
    rows_out = []
    for r in false_declared:
        key = (r["seed"], r["w"], r["t"])
        n_atk = attacks_made.get(key, 0)
        rows_out.append({"seed": r["seed"], "w": r["w"], "t": r["t"], "full_swing": bool(n_atk >= r["xs"]),
                         "attacks_made": n_atk, "xs": r["xs"], "n_block": r["blockers"],
                         "through": r["through"], "stops": r["stops"], "hits": r["hits"], "life": r["life"],
                         "counters_theory": r.get("counters"), "actual": defender_actual.get(key)})
    full = [r for r in rows_out if r["full_swing"]]
    with_actual = [r for r in full if r["actual"] is not None]
    return {"n_false_declared": len(rows_out), "n_full_swing": len(full),
            "n_full_swing_with_actual": len(with_actual),
            "full_swing_rows": full,
            # 平均（追えた行だけ）——理論の stops／through 対 実際の blocked／counters_used
            "means": {
                "stops": round(float(np.mean([r["stops"] for r in with_actual])), 3) if with_actual else None,
                "through": round(float(np.mean([r["through"] for r in with_actual])), 3) if with_actual else None,
                "n_block": round(float(np.mean([r["n_block"] for r in with_actual])), 3) if with_actual else None,
                "blocked_actual": round(float(np.mean([r["actual"]["blocked"] for r in with_actual])), 3) if with_actual else None,
                "counters_used_actual": round(float(np.mean([r["actual"]["counters_used"] for r in with_actual])), 3) if with_actual else None,
                "targeted_actual": round(float(np.mean([r["actual"]["targeted"] for r in with_actual])), 3) if with_actual else None,
            }}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--don", default="on", choices=("on", "off"))
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = collect(a.src, a.games, a.don == "on")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
