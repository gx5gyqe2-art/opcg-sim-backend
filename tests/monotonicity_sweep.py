"""計器② 単調性スイープ（dev・docs/reports/cpu_correctness_instruments_20260628.md §4）。

eval(L1) の**系統的な構造バグ**を、外部ラベル無しで自動検出する。
**無条件に得な摂動**（自キャラ power+1000／ライフ+1／手札+1）を局面へ加え、評価値 V が下がったら違反。
盤面追加（新キャラ展開）は過剰展開トリガ等で正当に非単調になり得るので**対象外**（偽陽性源）。
違反はハード失敗でなく**トリアージ候補**として抽出する。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/monotonicity_sweep.py --games 20 --max-plies 30
"""
import argparse
import copy
import random

import conftest  # noqa: F401
from cpu_selfplay import _load_db, build_deck
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, action_api

EPS = 1.0  # eval は ×V2_SCALE(=2000) 単位。これ未満の低下はノイズとみなす。


def _gen_snapshots(db, n_games, max_plies, every, seed0):
    """decide_guarded（pimc=1・小予算）で対局を進め、`every` ply ごとに盤面クローンを採取。"""
    snaps = []
    cpu_ai.set_budget_override(40)
    try:
        for g in range(n_games):
            seed = seed0 + g
            random.seed(seed)
            l1, c1 = build_deck(db, "p1")
            l2, c2 = build_deck(db, "p2")
            m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
            m.start_game()
            rng = random.Random(seed * 7 + 1)
            for ply in range(max_plies):
                if m.winner is not None:
                    break
                pend = m.get_pending_request()
                if not pend:
                    break
                pid = pend.get("player_id") or pend.get(list(pend.keys())[0])
                actor = m.p1 if m.p1.name == pid else m.p2
                if ply % every == 0:
                    snaps.append(m.clone())
                try:
                    mv = cpu_ai.decide_guarded(m, actor, "hard", rng, pimc_worlds=1)
                except Exception:
                    break
                if mv is None:
                    break
                try:
                    if mv["kind"] == "battle":
                        action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
                    else:
                        action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
                except Exception:
                    break
    finally:
        cpu_ai.set_budget_override(None)
    return snaps


def _perturb_power(snap, name):
    """自キャラ1体に power+1000（盤面枚数不変＝最も副作用の無い摂動）。適用不可なら None。"""
    me = snap.p1 if snap.p1.name == name else snap.p2
    if not me.field:
        return None
    c = snap.clone()
    cm = c.p1 if c.p1.name == name else c.p2
    cm.field[0].power_buff += 1000
    return c


def _perturb_zone(snap, name, zone):
    """life / hand に1枚追加（デッキ札の deepcopy＝デッキ枚数も減らさない純増）。"""
    me = snap.p1 if snap.p1.name == name else snap.p2
    src = (me.deck or me.hand or me.life)
    if not src:
        return None
    c = snap.clone()
    cm = c.p1 if c.p1.name == name else c.p2
    card = copy.deepcopy((cm.deck or cm.hand or cm.life)[0])
    getattr(cm, zone).append(card)
    return c


PERTURBS = {
    "power+1000": _perturb_power,
    "life+1": lambda s, n: _perturb_zone(s, n, "life"),
    "hand+1": lambda s, n: _perturb_zone(s, n, "hand"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=30)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    db = _load_db()
    snaps = _gen_snapshots(db, args.games, args.max_plies, args.every, args.seed0)
    print(f"局面数: {len(snaps)}（{args.games}局・{args.every}plyごと）")

    stats = {k: {"n": 0, "viol": 0, "min_delta": 0.0, "sum_delta": 0.0, "zero": 0, "samples": []} for k in PERTURBS}
    for idx, snap in enumerate(snaps):
        for name in (snap.p1.name, snap.p2.name):
            try:
                base = cpu_ai.evaluate(snap, name)
            except Exception:
                continue
            for pname, fn in PERTURBS.items():
                c = fn(snap, name)
                if c is None:
                    continue
                try:
                    v2 = cpu_ai.evaluate(c, name)
                except Exception:
                    continue
                delta = v2 - base
                st = stats[pname]
                st["n"] += 1
                st["sum_delta"] += delta
                if abs(delta) <= EPS:
                    st["zero"] += 1
                if delta < -EPS:
                    st["viol"] += 1
                    if delta < st["min_delta"]:
                        st["min_delta"] = delta
                    if len(st["samples"]) < 5:
                        st["samples"].append((idx, name, round(delta, 1)))

    print("\n=== 単調性スイープ結果（V(摂動後) − V(元) が負＝違反） ===")
    for pname, st in stats.items():
        n = st["n"] or 1
        rate = st["viol"] / n * 100
        zero = st["zero"] / n * 100
        mean = st["sum_delta"] / n
        print(f"  {pname:10s}: 検査{st['n']:5d}  違反{st['viol']:4d}({rate:5.2f}%)  "
              f"無反応{st['zero']:4d}({zero:5.1f}%)  平均Δ={mean:+.1f}  最悪Δ={st['min_delta']:.1f}")
        for s in st["samples"]:
            print(f"      - snap#{s[0]} side={s[1]} Δ={s[2]}")
    print("\n解釈: 違反>0 ＝ eval が『得な変化』で評価を下げる構造バグ。"
          "平均Δが+で無反応率が低い＝計器が機能（摂動が効いている）の健全性確認。")


if __name__ == "__main__":
    main()
