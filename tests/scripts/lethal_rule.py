#!/usr/bin/env python3
"""**決着の定義**（T136・2026-09-23・ユーザ指示「勝敗との紐付けは必須で…理論の側に決着の定義が必要かもね」→「それでお願いします。」）。
T117（「今このターンで殺せるか」・2026-09-19）を**決着の定義**として仕切り直したもの。

## 問い

理論には**決着（ここから先はどちらが勝つか決まっている）の定義が無かった**。交点の橋は「先に届いた側」を暗黙に
使い、線形の橋には無く、T117 の「殺せる判定」は**適合率 0.23**（桁で外れている唯一の場所）で止まっていた。
**勝敗の説明は、決着後（勝者は確定・「誰が・いつ」を当てる）と決着前（勝率で言う）に分かれる**——その境目がこの器。

## 定義（規則だけ・完全情報・新定数ゼロ）

> **ターン t の開始時に手番側 w が決着** ⇔ **相手が最善で守っても通る本数 ≥ 相手の残りライフ ＋ 1**。

**「＋1」は規則**（`rules/battle.rs`: **ライフが 0 のときに損害を受けたら負け**＝最後のライフ札を取られても負けではない）。
**T117 の判定（`≥ ライフ`）は 1 本足りない側に外れていた**うえ、**ライフ 0 の行（1 本通れば勝ち）を除外していた**。
本 T の `by_life` 表（ライフ 1 でドン込みの精度 0.89・そのターンに終わった精度は 0.46）がこれを露わにした。

* **通る本数** … 攻撃手（リーダー＋このターン攻撃できる体・`own_attackers_of`）のうち、
  相手の**アクティブなブロッカー**に横取りされない本数（ブロッカーは安い攻撃から止める＝守り手の最良）。
* **止められる本数** … **守り手の実際の手札**（§0.05・**相手席の直近の自席ターンの最後の行**から読む＝T130 と同じ規約）で、
  カウンター値が足りる組を作れる本数。**守り手は「切れるだけ切る」**（`max`）＝詰みの判定
  （守り手はライフ 0 なら経済を捨てて守る・規則）。**安い攻撃から止め、1 本ごとに使うカウンター値が最小の組を選ぶ**
  （本数を最大にする貪欲）。比較用の `econ`（T130 の `attacks_stopped`＝受けるより安いときだけ切る・切替
  `LETHAL_STOP_MODE`）は 2026-10-05 に削除——`claude/theory-switches-final` で再現できる。
* **守り手の手札** … 実際の手札（`actual`）。T117 の旧規約（`share`＝`相手の手札枚数 × デッキの切れる割合`・
  T117 の 0.23 を再現するための切替 `LETHAL_HAND_MODE`）は 2026-10-05 に削除——`claude/theory-switches-final` で再現できる。
* **ドン付与** … `--don on`（既定）で**アクティブなドンを安い攻撃から 1 体 4 枚まで**（規則）配って `x` を上げる。
* **カウンターのイベントはドンを払う**（規則）——守り手が使えるのは**相手ターン開始時に残しているアクティブなドン**
  （`sc[SC_OPP_DON_ACTIVE]`）の範囲まで。**イベントの費用の和がそれを超える組は選べない**。
* **守り手の手札の読みは正確**——「最後の判断行で出した札が残っているのでは」と疑って落とす処理を入れて数えたが
  **0 回**（ターン最後の判断行の選択は `TURN_END` で、その行の手札は出した後のもの）。処理は外した。
* **受けたライフの札** … `draw`＝**受けたライフの札は手札に入る**（規則・`rules/battle.rs` の `dest = Zone::Hand`）ので、
  守り手は**残りライフの枚数**（全部が手札に入ってから、次の損害で負ける）を**デッキの平均カウンター値**の札として追加で持つ
  （**完全情報でもライフの中身の順は読めない**のでデッキ平均＝T91 と同じ規約・新定数ゼロ）。**先に全部持たせる**のは
  守り手に有利な上限＝**宣言は健全側**（取りこぼしは増えうる）。旧の `off`（T117 と同じ・ライフの札を数えない・
  切替 `LETHAL_LIFE_MODE`）は 2026-10-05 に削除——`claude/theory-switches-final` で再現できる。

## 段 7（2026-10-07）: 判定は Rust だけ

判定（`lethal_of_row`・`attach_don`・`max_stops`・デッキの平均カウンター値）は Rust の局の駆動
（`rust/opcg_engine/src/theory/core/drv_lr.rs`）にある。Python に残るのは他の器が読む入口 `settled_map` だけ
（記録を読み・デッキを作り直して 1 局ずつ Rust に渡す）。適合率・再現率を測る T136 の CLI（`collect`・`declare_metrics`）は
Python の理論と一緒に消した——凍結ブランチ `claude/theory-switches-final` で再現する。
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import theory_rs as TR  # noqa: E402

#: 受けたライフの札の平均カウンター値の読み（規則どおり・定数・Rust の `leaves_deck::avg_counter`）
AVG_COUNTER_MODE = "rules"


def settled_map(dirs, limit_games=0, with_don=True):
    """**T138a**: 局×席×ターンの決着フラグ `{(seed, w, t): declared(bool)}`。**判定は `lethal_of_row` そのもの**
    （Rust の局の駆動 `lethal_rule`）——他の器（`win_calib`・`relative_ledger`・`crossing_bridge`）が
    「このターンより後は決着後」を読むための入口。**手札が読めない行（先手 1 ターン目）は `False`**。"""
    import deck_refill as DR
    import theory_bridge as TB
    decks = DR.decks_by_seed(dirs)
    if not decks:
        raise ValueError("デッキが引けない（受けたライフの札の平均カウンター値が読めない・%s）" % (dirs,))
    out = {}
    games = 0
    for game in PL.iter_games(dirs, row_cols=TB.ROW_COLS, pol_cols=TB.POL_COLS, extra_fn=TB._extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = TR.seed_of(game)
        dk = decks.get(seed) if decks else None
        pin = {"decks": TR.deck_list(dk)}
        res = TR.game_call("lethal_rule", game, {"cfg": TR.cfg(with_don=bool(with_don)), "in": pin, "stats": {}, "carry": {}},
                           counters=False)
        for w, t, dec in res["rows"]:
            out[(seed, w, t)] = bool(dec)
    return out
