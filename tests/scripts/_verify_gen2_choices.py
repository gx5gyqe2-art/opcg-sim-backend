"""検証①-②: 配線バグ修正後、出荷Gen2 が2シーンで「正しい方」を選ぶか。
各選択手を適用した結果状態を出荷Gen2の value net で採点し、net がどちらを好むかを見る
（＝1-ply で net が accept/decline・add/decline のどちらに高い価値を置くか）。
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import numpy as np
from engine_helpers import make_game, make_instance, make_master
from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.core import cpu_learned

cpu_learned._lazy_init()
vf = cpu_learned._value_fn(cpu_learned._STATE["vnet"], cpu_learned._STATE["vocab"])
G = OPCGGame()


def eval_move(gm, move, who):
    st2 = G.apply(gm, move, who)
    if st2 is None:
        return None
    return float(vf(st2, who))


print("=== シーン#2: OP16-119【登場時】上3枚から1枚までライフ追加 ===")
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
db = CardLoader(os.path.join(_ROOT, "opcg_sim", "data", "opcg_cards.json")); db.load()
teach = db.get_card("OP16-119")
onplay = [ab for ab in teach.abilities if ab.trigger.name == "ON_PLAY"][0]
gm, p1, _ = make_game()
p1.deck = [make_instance(make_master(card_id=f"D-{i}", cost=i + 1), owner=p1.name) for i in range(6)]
src = make_instance(teach, owner=p1.name)
gm.resolve_ability(p1, onplay, source_card=src)
assert gm.active_interaction["action_type"] == "SELECT_TARGET"
life_before = len(p1.life)
moves = G.legal_actions(gm)
print(f"  ライフ枚数(選択前)={life_before}  列挙手数={len(moves)}")
best = None
for mv in moves:
    su = mv["payload"].get("selected_uuids") or []
    v = eval_move(gm, mv, p1.name)
    tag = f"追加{len(su)}枚" if su else "見送り(0枚)"
    print(f"    {tag:14s} → net価値={v:+.4f}" if v is not None else f"    {tag}: apply失敗")
    if v is not None and (best is None or v > best[0]):
        best = (v, "追加" if su else "見送り")
print(f"  → net が選ぶ: {best[1]}（価値{best[0]:+.4f}）  期待=追加")
