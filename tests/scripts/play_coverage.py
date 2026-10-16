"""**エンジンはその手を打てるのか**（能力の検査・棋譜の頻度ではない・読み取り専用）。

ユーザ指摘 2026-09-14「**打たれた局からピックアップするのではなく、エンジンとして
プレイできるかどうかで定量化した方が良いんじゃない？**」。

## なぜ頻度では足りないか

**「60 局で 1 度も打たれなかった」は頻度の証拠であって、能力の証拠ではない**
——その盤面が来なかっただけかもしれない。`measurement.md` §1 の
「**〜しないではなく〜できない**」を、**行動空間そのもの**に当てはめた形である。

**実害が出た**（2026-09-14）: 棋譜からの数え上げで「**エンジンはステージを出せない**」と
報告したが、**本器で直接訊いたら出せた**。棋譜で 0 だったのは別の理由だった（下記）。

## 測り方——**エンジンに直接訊く**

```
そのカード 50 枚のデッキを作る → 手札に必ず来る → `get_legal_actions` に PLAY が出るか
```

**契機の交絡も外す**——**`COUNTER`／`TRIGGER` だけのイベントはメインで打てないのが正解**なので、
判定の母数から外す（**2026-09-14 に外し忘れて「打てない」と誤報した**）。

**コストの交絡を外す**のが肝——**種別ごとに最安のカードで揃える**。
コスト 2 のステージがターン 1 で打てないのは種別のせいではなくドンのせいなので、
**同じコスト帯で比べないと「出せない」と誤診する**。

## 3 つの層を分ける（ここが本器の眼目）

| 層 | 何を見るか | 見方 |
|---|---|---|
| **① 合法か** | ルール上打てるか | **本器**（`get_legal_actions`） |
| **② 探索の候補に在るか** | 枝刈りを通ったか | 記録の `pol_*` |
| **③ 選ばれるか** | 評価が選んだか | 記録の `pol_chosen` |

**①が × なら②③では代替できない**（`cpu_theory_gap.md` §6）。
**①が ○ で②が ×** なら**枝刈りの問題**——評価をいくら直しても打たれない。
**①②が ○ で③が ×** なら**評価・方策の問題**＝価格の話になる。

**この 3 層を混ぜると、手当ての場所を間違える。**

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/play_coverage.py --out ~/play_cov.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: 素のリーダーと素のキャラ（相手のデッキを一定に保つため）
VANILLA_LEADER = "OP01-001"
VANILLA_CHAR = "OP01-016"
#: 検査する種別
KINDS = ("char", "event", "stage")
#: 手札に来ないと検査にならないので、そのカードだけでデッキを組む
DECK_SIZE = 50
#: **メインフェイズに打てるはずの契機**。これを持たないイベント（カウンター専用）は
#: **メインで出ないのが正解**なので、欠陥として数えてはいけない（2026-09-14 に誤報した）。
MAIN_TRIGGERS = ("ON_PLAY", "ACTIVATE_MAIN", "MAIN", "PASSIVE", "RULE", "UNKNOWN", None)


def main_playable_expected(kind, abilities):
    """**そのカードはメインフェイズに打てるはずか**（契機から決める）。

    キャラとステージは場に残る常設なので常に打てるはず。**イベントは契機に依る**
    ——`COUNTER`／`TRIGGER` だけのイベントは**相手ターンの窓でしか打てない**ので、
    メインで出ないのは**正しい挙動**である。
    """
    if kind in ("char", "stage"):
        return True
    trig = [(ab.get("trigger") or ab.get("timing")) for ab in (abilities or [])]
    if not trig:
        return True
    return any((t in MAIN_TRIGGERS) for t in trig)


def kind_of(info):
    # **空の dict は「フラグが無いキャラ」**で、`None`（カードが判らない）とは別物。
    # `if not info` で潰すとキャラが丸ごと種別不明になる（2026-09-14 にテストが捕まえた）。
    if info is None:
        return None
    if info.get("leader"):
        return "leader"
    if info.get("stage"):
        return "stage"
    if info.get("event"):
        return "event"
    return "char"


def cheapest_by_kind(cards, card_ids):
    """**種別ごとの最安カード**（コストの交絡を外すため）。"""
    best = {}
    for cid in card_ids:
        info = cards.info(cid)
        k = kind_of(info)
        if k not in KINDS:
            continue
        c = info.get("cost")
        if c is None:
            continue
        if k not in best or c < best[k][1]:
            best[k] = (cid, int(c))
    return best


def _keep_hands(game):
    """マリガンの窓を片づけて MAIN_ACTION まで進める。"""
    for _ in range(4):
        pend = game.get_pending_request(False)
        if not pend or pend.get("action") != "MULLIGAN":
            break
        game.apply_game_action(pend["player_id"], "KEEP_HAND", {})


def probe_play(cid, leader=VANILLA_LEADER, opp_card=VANILLA_CHAR):
    """**そのカードを PLAY できるか**をエンジンに直接訊く。"""
    from opcg_sim.api.engine_rs import RsGame
    game = RsGame.create_from_ids("P1", "P2", leader, [cid] * DECK_SIZE,
                                  leader, [opp_card] * DECK_SIZE, first_player="p1")
    _keep_hands(game)
    board = game.board()
    p1 = board["players"]["p1"]
    hand = p1["zones"]["hand"]
    uuids = {c["uuid"] for c in hand if c["card_id"] == cid}
    legal = game.get_legal_actions("P1")
    plays = [m for m in legal if m.get("action_type") == "PLAY"
             and (m.get("payload") or {}).get("uuid") in uuids]
    return {"in_hand": len(uuids), "legal_total": len(legal),
            "play_offered": len(plays),
            "playable": bool(plays),
            "action_types": sorted({m.get("action_type") for m in legal})}


def run(cards, card_ids, limit_per_kind=3, raw=None):
    """種別ごとに**最安から数枚**を検査する（1 枚では偶然を排除できない）。

    **メインで打てるはずでないカードは判定の母数から外す**（カウンター専用イベント）。
    """
    by_kind = {}
    for cid in card_ids:
        info = cards.info(cid)
        k = kind_of(info)
        if k not in KINDS or info.get("cost") is None:
            continue
        by_kind.setdefault(k, []).append((int(info["cost"]), cid))
    out = {}
    for k, lst in by_kind.items():
        lst.sort()
        rows = []
        for cost, cid in lst[:limit_per_kind]:
            abils = ((raw or {}).get(cid) or {}).get("abilities") or []
            got = probe_play(cid)
            got.update(card_id=cid, cost=cost,
                       expected=main_playable_expected(k, abils),
                       triggers=sorted({str(ab.get("trigger") or ab.get("timing"))
                                        for ab in abils}))
            rows.append(got)
        scoped = [r for r in rows if r["expected"]]
        out[k] = {"tested": rows, "in_scope": len(scoped),
                  "out_of_scope": len(rows) - len(scoped),
                  "playable_share": (round(sum(1 for r in scoped if r["playable"])
                                           / len(scoped), 4) if scoped else None)}
    return out


def verdict(res):
    """**①合法かの層の判定**——打てない種別が在れば、そこは評価では直せない。"""
    bad = [k for k, v in res.items() if v.get("playable_share") == 0.0]
    if bad:
        return {"verdict": "not_legal_for_some_kinds", "kinds": sorted(bad)}
    return {"verdict": "all_kinds_legal"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--effects", default="")
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(_ROOT, "tests"))
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    path = a.effects or os.path.join(_ROOT, "opcg_sim", "data", "opcg_effects.json")
    with open(path, encoding="utf-8") as fh:
        ids = list(json.load(fh)["cards"])

    t0 = time.time()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)["cards"]
    res = run(cards, ids, a.per_kind, raw)
    pool = Counter(kind_of(cards.info(cid)) for cid in ids)
    out = {"pool": dict(pool), "by_kind": res}
    out.update(verdict(res))
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
