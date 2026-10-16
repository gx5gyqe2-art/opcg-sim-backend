#!/usr/bin/env python3
"""**値段の付いていない遷移を数える**（T123・`H`・2026-09-20・`game_theory.md` §17.9.0 の残り 1）。

## 問い

紐付けの法則（§17.9）は **`Σ ΔW = W_end − W₀`**（telescoping）1 本で水準と帰属を結ぶ。
**ところがこの等式は閉じていない**——帳簿は**自席の手**しか積まないが、`W` は**全部の遷移**で動く:

* **引き**（毎ターン 1 枚・規則）
* **【トリガー】とライフ→手札**（受けた 1 枚が手札に入る＝`h = 0.89` の出どころ）
* **ターンの境目**（ドン +2・アンタップ・召喚酔いの解除）——**誰も「手を打っていない」のに時計が動く**
* 効果の遅延解決・KO の後始末

**閉じない限り、判別と較正は原理的に両立しない**（T122 §5 の論証）:
正しい `ΔW` は決着したら黙るので、局ごとの和を勝敗で採点する指標では不利になる。
`Σ ΔW = z − W₀` が厳密なら、静かな帳簿でも合計はちょうど勝敗を当てる。

**だから本器の問いは「どれを値付けすれば閉じるか」**であって、「閉じるか」ではない（閉じるのは恒等式）。

## 測り方（当てはめゼロ・恒等式の配分だけ）

1. 全部の判断行を**記録の順**に並べ、**席 0 の視点**で `W` を出す。
2. 隣り合う行の差 `W_{i+1} − W_i` を**そのまま**取る（**和は `W_last − W_first` に厳密に一致する**＝検算）。
3. その差を 2 つに割る: **`priced`**＝行 `i` で選ばれた手が説明する分（`W(手を当てた状態) − W(状態)`）と
   **`residual`**＝残り。
4. `residual` を **5 つの軸（`Θ_me`・`Θ_opp`・`A_me`・`A_opp`・自席ターン番号 `j`）へシャープレイ値で配る**
   ——**順序に依らず、和が厳密に差に一致する**唯一の配り方（部分集合 2⁵ = 32 通りの評価で出る）。
5. **ターンの境目を跨ぐ差**と**同じターンの中の差**を分けて数える（規則の出どころが違う）。

**新定数ゼロ**（配分は恒等式・`σ_rel` は既測）。**「当てはめて残差を説明する係数」は 1 つも置かない**。

使い方:

    python tests/scripts/transition_ledger.py --in <records_dir> [--games N] [--json out.json]
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
import crossing_bridge as CB  # noqa: E402
import kappa_vector as KV  # noqa: E402
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

#: 配分する軸（`j` も軸に入れる——輪郭の読み出し位置が進むのは**誰の手でもない遷移**）
AXES5 = ("th_me", "th_opp", "a_me", "a_opp", "j")
#: `residual` を規則の出どころで分ける区分
CAUSES = ("turn_boundary", "same_turn")

#: **ターンの境目に値段を付けるか**（T124／T125）——引き 1 枚・アンタップ・ドン +2 ＋ 召喚酔いの解除を規則から値付けする
#: 切替 `BOUNDARY_MODE`（`rules`／`draw_untap`／`draw`／`untap`／`don`）は「境目に速さの項は当てられていない」
#: （T125・`2026-09-20_boundary_don_retraction.md`）で引っ込め、2026-10-05 に削除——`claude/theory-switches-final` で
#: 再現できる。境目は値付けしない（`off`・T123 の測り方）。出力の `boundary_mode` は定数 `"off"`。


def _priority(acc):
    """**残差を 3 つに割る**（全部 `|·|` の割合・手当てが別々なので分ける）。

    同じターンの中の残差を**攻撃の行**と**攻撃でない行**に分け、そこへ**ターンの境目**を並べる。3 つで 1 になる。
    **`curve` の読みでは「攻撃でない行」の価格は厳密に 0**（速さの軸が無い）だが、`theory` では価格が付く
    ——だから名前は「値段が付いていない」ではなく「攻撃でない」にしてある（読みによって意味が変わる欄にしない）。"""
    tot = max(1e-12, acc["resid_abs"])
    st_attack = acc["fam_abs"].get("attack", 0.0)
    st_other = sum(v for f, v in acc["fam_abs"].items() if f != "attack")
    return {"attack_rows": round(st_attack / tot, 4),
            "nonattack_rows": round(st_other / tot, 4),
            "turn_boundary": round(acc["by_cause_abs"]["turn_boundary"] / tot, 4)}



def collect(dirs, limit_games=0, theta=THETA, mu=MU):
    """記録を 1 度読んで **`W` の差を `priced` と `residual` に割り、`residual` を軸と区分へ配る**。
    1 局ぶんの計算（`W` の差・シャープレイ値・区分）は Rust の局の駆動（`core::drv_tl`・`acc` は局をまたいで渡す）。"""
    prof = CB.profile_for(dirs)
    if not prof:
        raise ValueError("D_MODE=KV.D_MODE なのに損害の輪郭が引けない（%s）" % (dirs,))
    sr = CB.sigma_rel_for(dirs, slope="curve")
    if sr is None:
        raise ValueError("σ_rel が引けない＝黙って別の物差しに落とさない（T118 の規約）")
    TR.set_sigma_rel(sr)
    seat_decks = KV._seat_decks(dirs)     # **T128**: `A` の流入・効果はデッキの中身から出る
    acc = {"gap": 0.0, "gap_abs": 0.0, "priced": 0.0, "priced_abs": 0.0,
           "resid": 0.0, "resid_abs": 0.0,
           "by_axis": {a: 0.0 for a in AXES5}, "by_axis_abs": {a: 0.0 for a in AXES5},
           "by_cause": {c: 0.0 for c in CAUSES}, "by_cause_abs": {c: 0.0 for c in CAUSES},
           # **原因 × 軸**（ターンの境目で動く軸と、同じターンの中で動く軸は規則の出どころが違う）
           "cross_abs": {c: {a: 0.0 for a in AXES5} for c in CAUSES},
           "cross_n": {c: 0 for c in CAUSES},
           # **行の手の型ごとの残差**——`same_turn` の残差は
           # 「相手が窓で答えた（攻撃の行に偏る）」と「価格が合っていない（出す・効果の行に偏る）」の
           # **2 つが混ざっている**ので、型で割らないと切り分けられない（P8 が開くまでこれが最良の分離）。
           "fam_abs": {}, "fam_n": {}, "fam_priced_abs": {}}
    stats = {"games": 0, "rows": 0, "gaps": 0, "identity_max_err": 0.0,
             "w_first_sum": 0.0, "w_last_sum": 0.0, "z_sum": 0.0, "terminal_sum": 0.0,
             "terminal_abs_sum": 0.0}
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        pin = {"decks": TR.deck_list(KV._deck_pair(seat_decks, TR.seed_of(game)))}
        res = TR.game_call("transition_ledger", game, {"cfg": TR.cfg(theta, mu, prof=prof, sr=float(sr)), "in": pin,
                                                       "stats": stats, "carry": {"acc": acc}})
        new_acc = res["carry"]["acc"]
        acc.clear()
        acc.update(new_acc)
        new = res["stats"]
        stats.clear()
        stats.update(new)
    n = max(1, stats["gaps"]); ng = max(1, stats["games"])
    tot_abs = max(1e-12, acc["gap_abs"])
    out = {"games": stats["games"], "rows": stats["rows"], "gaps": stats["gaps"],
           "d_mode": KV.D_MODE, "boundary_mode": "off", "sigma_rel": round(sr, 4),
           "attack_rest_mode": KV.ATTACK_REST_MODE,
           # **恒等式の検算**（配分の和が差に一致すること・telescoping が閉じること）
           "identity_max_abs_error": round(stats["identity_max_err"], 12),
           "w_first_mean": round(stats["w_first_sum"] / ng, 4),
           "w_last_mean": round(stats["w_last_sum"] / ng, 4),
           "z_mean": round(stats["z_sum"] / ng, 4),
           # **末端の隙間**（最後の行の `W` と実際の勝敗の差＝どの遷移にも載らない分）
           "terminal_mean": round(stats["terminal_sum"] / ng, 4),
           "terminal_abs_mean": round(stats["terminal_abs_sum"] / ng, 4),
           # **本題**: 差のうち手が説明する分と残り（絶対値の割合で読む＝符号の打ち消しを避ける）
           "priced_share": round(acc["priced_abs"] / tot_abs, 4),
           "resid_share": round(acc["resid_abs"] / tot_abs, 4),
           "priced_mean": round(acc["priced"] / n, 6), "resid_mean": round(acc["resid"] / n, 6),
           "by_axis_share": {a: round(acc["by_axis_abs"][a] / max(1e-12, sum(acc["by_axis_abs"].values())), 4)
                             for a in AXES5},
           "by_axis_mean": {a: round(acc["by_axis"][a] / n, 6) for a in AXES5},
           "by_cause_share": {c: round(acc["by_cause_abs"][c] / max(1e-12, acc["resid_abs"]), 4)
                              for c in CAUSES},
           "by_cause_mean": {c: round(acc["by_cause"][c] / n, 6) for c in CAUSES},
           "by_cause_gaps": {c: acc["cross_n"][c] for c in CAUSES},
           # **1 つの隙間あたりの残差**（割合ではなく密度で読む＝どの遷移が濃いか）
           "resid_abs_per_gap": {c: round(acc["by_cause_abs"][c] / max(1, acc["cross_n"][c]), 6)
                                 for c in CAUSES},
           # **同じターンの中の残差を手の型で割る**（相手の窓の答え 対 価格の誤り）
           "by_family": {f: {"gaps": acc["fam_n"][f],
                             "resid_per_gap": round(acc["fam_abs"][f] / max(1, acc["fam_n"][f]), 6),
                             "priced_per_gap": round(acc["fam_priced_abs"].get(f, 0.0)
                                                     / max(1, acc["fam_n"][f]), 6),
                             "resid_over_priced": (round(acc["fam_abs"][f]
                                                         / acc["fam_priced_abs"][f], 3)
                                                   if acc["fam_priced_abs"].get(f) else None)}
                         for f in sorted(acc["fam_n"], key=lambda x: -acc["fam_n"][x])},
           # **残差の優先順位**（3 つに割る・各々に別の手当てが要る）:
           # `attack_rows`＝攻撃の行の説明できない分（**相手の窓の答え**と**価格の誤り**が混ざる＝P8 待ち）／
           # `nonattack_rows`＝**攻撃でない行**（体を出す・付与・`TURN_END`・効果）
           #   ——**`curve` の読みではここの価格が厳密に 0**（速さの軸が無いので）。`theory` では価格が付く／
           # `turn_boundary`＝**誰も手を打っていない遷移**（ドン +2・アンタップ・召喚酔いの解除・引き）。
           "resid_priority": _priority(acc),
           # **原因 × 軸**（各原因の中での軸の割合）
           "cross_share": {c: {a: round(acc["cross_abs"][c][a]
                                        / max(1e-12, sum(acc["cross_abs"][c].values())), 4)
                               for a in AXES5} for c in CAUSES}}
    # **N-3**（値段の窓は `joint` だけ）
    out["cut_price"] = {"mode": TR.SW["CUT_PRICE_MODE"],
                        **{k: v for k, v in stats.items() if str(k).startswith("cut_")}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="値段の付いていない遷移を数える（T123・H）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
