"""検証①-#2(実戦・決着用): 出荷Gen2が黒ひげ実戦で
(a) OP16-119 を実際にプレイするか、(b) その ON_PLAY ライフ追加選択で「追加」を選ぶか、
(c) learned側のライフが効果で実際に増える瞬間があるか、を追う。
"""
import argparse
import random
import numpy as np
import conftest  # noqa: F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, cpu_learned

DECK = "blackbeard_black_yellow"


def played_card_id(m, mv):
    """PLAY 手が置くカードの card_id を返す（分からなければ None）。"""
    pl = (mv or {}).get("payload") or {}
    uid = pl.get("uuid") or mv.get("card_uuid")
    if not uid:
        return None
    try:
        c = m._find_card_by_uuid(uid)
    except Exception:
        return None
    mm = getattr(c, "master", None)
    return getattr(mm, "card_id", None) if mm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    db = _load_db()
    l1rng = random.Random(args.seed + 777)
    nrng = np.random.default_rng(args.seed)

    plays_119 = 0                 # learned が OP16-119 をプレイした回数
    life_selection_add = 0        # OP16-119由来 SELECT_TARGET で「追加」を選んだ回数
    life_selection_decline = 0    # 同「見送り」を選んだ回数
    life_gain_events = 0          # learned のライフが+1した瞬間の回数

    for pair in range(args.pairs):
        for seat in ("p1", "p2"):
            _l1, c1 = HD.build(db, DECK, "p1"); _l2, c2 = HD.build(db, DECK, "p2")
            m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
            me = m.p1 if seat == "p1" else m.p2
            prev_life = len(me.life)
            ply = 0
            while ply < args.ply_cap and m.winner is None:
                pa = m.pending_actor_action()
                if pa is None:
                    break
                nm = pa[0]
                actor = m.p1 if m.p1.name == nm else m.p2
                if nm == seat:
                    pend = getattr(m, "active_interaction", None)
                    at = pend.get("action_type") if pend else None
                    src = pend.get("source_card_name") if pend else None
                    mv = cpu_learned.decide_learned(m, actor, sims=args.sims, rng=nrng)
                    if mv is not None:
                        if mv.get("action_type") in ("PLAY", "PLAY_CHARACTER") or mv.get("kind") == "play":
                            if played_card_id(m, mv) == "OP16-119":
                                plays_119 += 1
                        if at == "SELECT_TARGET" and src == "マーシャル・D・ティーチ":
                            pl = mv.get("payload") or {}
                            if pl.get("selected_uuids"):
                                life_selection_add += 1
                            else:
                                life_selection_decline += 1
                else:
                    mv = cpu_ai.decide(m, actor, rng=l1rng, info_policy="fair", pimc_worlds=args.pimc)
                if mv is None:
                    break
                try:
                    cpu_ai._apply_move_inplace(m, nm, mv)
                except Exception:
                    break
                cur = len(me.life)
                if cur > prev_life:
                    life_gain_events += 1
                prev_life = cur
                ply += 1

    print(f"\n=== 出荷Gen2 実戦 {args.pairs}ペア（黒ひげ・sims{args.sims}） ===")
    print(f"  learned が OP16-119 をプレイした回数 = {plays_119}")
    print(f"  OP16-119由来 ライフ追加選択: 追加={life_selection_add} / 見送り={life_selection_decline}")
    print(f"  learned のライフが+1した瞬間（効果でライフ増）= {life_gain_events}")


if __name__ == "__main__":
    main()
