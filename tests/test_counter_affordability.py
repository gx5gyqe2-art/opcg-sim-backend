"""カウンター合法手の支払い可能性フィルタの回帰テスト（副産物バグ修正）。

バグ: SELECT_COUNTER の候補生成（`engine/interaction.py`）が**イベントカウンターの発動コストを
支払えるか**を検査せず、active ドン!! 不足でも合法手に出していた。選ぶと `apply_counter` → `pay_cost` で
「ドン!!が不足」例外＝クラッシュ。MAIN_ACTION の PLAY は支払い可能性でフィルタしているのに非対称だった。

修正: イベントカウンターは `master.cost <= len(don_active)` の分だけ提示する。

本テストは実デッキ（イベントカウンターを含む）× ランダム自己対戦で、**全 SELECT_COUNTER 局面で
提示されるイベントカウンターが必ず支払い可能**であること・**ドン不足クラッシュが起きない**ことを固定する
（property ベース＝バグ再混入を実軌跡で検出）。ランダム方策は高速。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import game_driver as GD
import heldout_decks as HD
from opcg_sim.src.models.enums import CardType, TriggerType


class _CounterAffordabilityObserver:
    def __init__(self):
        self.unaffordable = []   # (turn, card_id, cost, don_active)

    def on_decision_point(self, ctx):
        pend = ctx.pending
        if not pend or pend.get("action") != "SELECT_COUNTER":
            return
        actor = ctx.actor
        don_active = len(actor.don_active)
        for mv in ctx.manager.get_legal_actions(actor):
            uid = mv.get("card_uuid")
            if not uid:
                continue
            c = next((x for x in actor.hand if x.uuid == uid), None)
            if c is None:
                continue
            if c.master.type == CardType.EVENT and any(
                    a.trigger == TriggerType.COUNTER for a in c.master.abilities):
                if (c.master.cost or 0) > don_active:
                    self.unaffordable.append((ctx.turn, c.master.card_id, c.master.cost, don_active))


def test_counter_list_never_offers_unaffordable_event_counter():
    """実デッキ×ランダム自己対戦: SELECT_COUNTER が払えないイベントカウンターを出さない＋クラッシュ無し。"""
    db = GD.load_db()
    ids = HD.deck_ids()

    def deckb(_db, seed):
        l1, c1 = HD.build(_db, ids[seed % len(ids)], "p1")
        l2, c2 = HD.build(_db, ids[(seed + 1) % len(ids)], "p2")
        return l1, c1, l2, c2

    obs = _CounterAffordabilityObserver()
    crashes = []
    for seed in range(20):
        seats = {pid: GD.make_seat(kind="random") for pid in ("p1", "p2")}
        try:
            GD.run_game(seed, db, seats=seats, observers=[obs], legal_moves="check",
                        deck_builder=deckb)
        except Exception as e:  # noqa: BLE001  ドン不足等のクラッシュを可視化
            crashes.append(f"seed={seed} {type(e).__name__}: {e}")

    assert not obs.unaffordable, (
        f"払えないイベントカウンターが合法手に出た: {obs.unaffordable[:5]}")
    assert not crashes, f"ランダム自己対戦がクラッシュ: {crashes[:3]}"
