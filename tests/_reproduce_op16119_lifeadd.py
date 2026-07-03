"""検証①-#2(ログ状況再現・決着用): OP16-119【登場時】「上3枚を見て1枚までライフ追加」で、
出荷Gen2 が実際に「追加」を選ぶか確認する（旧バグ=構造的に絶対追加しなかった）。

富んだ盤面（leader_test_helpers.build=ライフ5・デッキ20）で ON_PLAY を発火。
先の診断で、この盤面では resolve_interaction も MCTS探索経路(OPCGGame.apply)も
ライフ追加を正しく完遂すること（5→6）を確認済み＝探索は追加の価値を見られる。
"""
import conftest  # noqa: F401
import numpy as np
from leader_test_helpers import build
from engine_helpers import make_instance
from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.core import cpu_learned

_DB = CardLoader("/home/user/opcg-sim-backend/opcg_sim/data/opcg_cards.json"); _DB.load()
_TEACH = _DB.get_card("OP16-119")
_ONPLAY = [a for a in _TEACH.abilities if a.trigger.name == "ON_PLAY"][0]


def setup():
    gm, p1, p2, L = build("OP16-080")           # 富んだ盤面（ライフ5・デッキ20）
    src = make_instance(_TEACH, owner=p1.name); p1.field.append(src)
    gm.resolve_ability(p1, _ONPLAY, source_card=src)   # ON_PLAY 発火 → SELECT_TARGET
    return gm, p1


def main():
    gm, p1 = setup()
    ai = gm.active_interaction
    print(f"発火した interaction = {ai and ai.get('action_type')}  source = {ai and ai.get('source_card_name')}")
    print(f"p1 ライフ(選択前) = {len(p1.life)}  上3候補 = {len(ai.get('candidates', []))}枚")
    moves = OPCGGame().legal_actions(gm)
    n_add = sum(1 for m in moves if (m.get('payload') or {}).get('selected_uuids'))
    n_dec = sum(1 for m in moves if not (m.get('payload') or {}).get('selected_uuids'))
    print(f"提示された選択肢: 追加={n_add} / 見送り={n_dec}   （両方>0＝配線修正済み）")

    # value net が「追加 vs 見送り」でどちらを高く見るか（1-ply）
    G = OPCGGame(); vf = cpu_learned._value_fn(cpu_learned._STATE["vnet"], cpu_learned._STATE["vocab"]) \
        if cpu_learned._STATE else None
    cpu_learned._lazy_init()
    vf = cpu_learned._value_fn(cpu_learned._STATE["vnet"], cpu_learned._STATE["vocab"])
    for m in moves:
        su = (m.get("payload") or {}).get("selected_uuids") or []
        st2 = G.apply(gm, m, p1.name)
        v = float(vf(st2, p1.name)) if st2 is not None else None
        print(f"    {'追加' if su else '見送り'} → net={None if v is None else round(v,4)}")

    # 出荷Gen2 に決めさせる（複数シード）
    results = []
    for s in range(5):
        gm2, _ = setup()
        mv = cpu_learned.decide_learned(gm2, gm2.p1, sims=160, rng=np.random.default_rng(s))
        su = (mv.get("payload") or {}).get("selected_uuids") if mv else None
        results.append(bool(su))
    n_add_gen2 = sum(1 for a in results if a)
    print(f"\n出荷Gen2 の選択(5seed) 追加した = {results}")
    print(f"  → 追加={n_add_gen2}/5  見送り={5 - n_add_gen2}/5   期待: 追加（ライフ増は生存に有利）")
    print("  " + ("✅ #2 ログ状況でライフ追加を選ぶ＝治っている" if n_add_gen2 >= 3
                  else "⚠ まだ見送り寄り＝netがライフ追加の価値を選好しない"))


if __name__ == "__main__":
    main()
