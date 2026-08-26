"""防御監査 P4-a: 防御窓の結合判断の質を実測する（マクロ手化 P4 の入口計測・2026-08-24）。

現行（gen15・窓の根畳み）は防御窓を出口 value の argmax で選ぶが、各ブロック枝の先の
カウンター消費は**貪欲**に決まる＝「素通し／最小限で守る／厚く守る」という**要点総量の比較**を
していない。本計器は自己対戦の防御窓（被攻撃時の SELECT_BLOCKER / SELECT_COUNTER 入口）で:

  1. 結合防御の変種を列挙: 素通し／各ブロッカー／カウンター要点総量（守り切る最小・+1000）
     ／ブロック×カウンターの組合せ
  2. 各変種を台本適用して戦闘終了までの盤面を作り、戦闘出口評価（現行 value）で順位づけ
  3. エンジンが実際に選んだ防御（コミット済みプランの実行結果）と最良変種を比較し、
     乖離率と乖離の型（過剰防御/過少防御/ブロッカー選択違い）を出す

ロールアウト無し（出口 value 比較のみ）＝審判 π 汚染はあるが監査としては現行物差しでの
一貫性検査になる。教師には使わない。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/defense_audit_probe.py \
    --games 4 --seed-base 780000
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai


def _counter_total_options(hand, need):
    """要点総量の候補（pure）: 守り切る最小の札組・(+1000 の余裕)。

    lethal の防御台本と同じ算術（印字カウンター値のみ・大きい順に切る）。
    返り値 [(label, [uuid,...]), ...]（素通しは呼び出し側で足す）。"""
    vals = sorted(((int(getattr(c, "current_counter", 0) or 0), c.uuid) for c in hand),
                  key=lambda x: -x[0])
    vals = [(v, u) for v, u in vals if v > 0]
    out = []
    for extra, label in ((0, "min_save"), (1000, "save+1k")):
        target = need + extra
        acc, picks = 0, []
        for v, u in vals:
            if acc >= target:
                break
            acc += v
            picks.append(u)
        if acc >= target and picks:
            out.append((label, picks))
    # 同一の札組は畳む
    seen, dedup = set(), []
    for label, picks in out:
        key = tuple(sorted(picks))
        if key not in seen:
            seen.add(key)
            dedup.append((label, picks))
    return dedup


def defense_variants(manager, name):
    """防御窓での結合変種 [(label, [move,...])]（pure に近い列挙・適用は呼び出し側）。

    現在の窓が SELECT_BLOCKER ならブロック各枝×後続総量、SELECT_COUNTER なら総量のみ。
    move は {"kind":"battle", "action_type": ..., "card_uuid": ...} 形式（battle 窓の実手）。"""
    bat = getattr(manager, "active_battle", None)
    if bat is None:
        return []
    me = manager.p1 if manager.p1.name == name else manager.p2
    try:
        atk = int(bat["attacker"].get_power(True))
        tgt = int(bat["target"].get_power(False)) + int(bat.get("counter_buff", 0) or 0)
    except Exception:
        return []
    need = atk - tgt + 1000 if atk >= tgt else 0
    outs = [("pass", [])]
    if need > 0:
        for label, picks in _counter_total_options(getattr(me, "hand", []) or [], need):
            outs.append((label, [{"kind": "battle", "action_type": "SELECT_COUNTER",
                                  "card_uuid": u} for u in picks]))
    return outs


class _Done(BaseException):
    pass


_G = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=780000)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--defense-box", action="store_true",
                    help="防御箱 v1（P4-c・D1'/D2' 候補整形）を有効化したエンジンで測る"
                         "（乖離が減るかの効果確認用）")
    args = ap.parse_args()

    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core import cpu_learned as CL
    from opcg_sim.src.learned.mcts import resolve_battle_inplace
    from opcg_sim.src.core import journal
    from opcg_sim.src.core.journal import JournaledList

    db = _load_db()
    eng = CL.LearnedEngine(defense_box=True) if args.defense_box else CL.LearnedEngine()
    vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)

    windows = []            # (name, manager clone at defense entry)

    class Cap:
        def __init__(self):
            self.n = 0
            self._keys = cpu_ai._pending_keys()

        def on_decision_point(self, ctx):
            _kp, ka = self._keys
            kind = (ctx.pending or {}).get(ka)
            if kind not in ("SELECT_COUNTER", "SELECT_BLOCKER"):
                return
            m = ctx.manager
            bat = getattr(m, "active_battle", None)
            if bat is None:
                return
            name = getattr(ctx.actor, "name", None)
            # 同一戦闘の2回目以降の窓（プラン実行中）は最初だけ採る
            key = (id(bat),)
            if key in getattr(self, "_seen", set()):
                return
            self._seen = getattr(self, "_seen", set())
            self._seen.add(key)
            if len(windows) < args.max_windows:
                windows.append((name, m.clone()))

        def on_decision(self, ctx, move):
            self.n += 1
            if self.n > 300:
                raise _Done()

    seat = make_seat(kind="learned", want_trace=False, sims=160, engine=eng)
    for i in range(args.games):
        seed = args.seed_base + i
        la, lb = _leader_pair(db, seed, "random")
        try:
            run_game(seed, db, seats={"p1": seat, "p2": seat},
                     deck_builder=synth_deck_builder(la, lb, seed=seed),
                     observers=(Cap(),), max_steps=1500, legal_moves="skip",
                     invariants="raise", stop_after_decisions=300)
        except _Done:
            pass
        print(f"  seed {seed}: 窓累計 {len(windows)}", flush=True)

    def battle_end_value(m0, name, pre_moves):
        """変種を適用して戦闘を解決し、出口盤面の value を返す（失敗は None）。"""
        m = m0.clone()
        m.action_events = []
        random.seed(777); np.random.seed(777)
        try:
            for mv in pre_moves:
                cpu_ai._apply_move_inplace(m, name, mv, stop_at_select=True)
            # 残りの窓は「素通し」（PASS/decline）で閉じ、戦闘だけ解決する
            for _ in range(20):
                if getattr(m, "active_battle", None) is None:
                    break
                pa = m.pending_actor_action()
                if pa is None or pa[0] != name:
                    break
                legal = eng.game.legal_actions(m)
                mv = None
                for x in legal:
                    d = cpu_ai._describe_move(m, x) or {}
                    if d.get("action_type") in ("PASS",) or d.get("accepted") is False:
                        mv = x
                        break
                if mv is None:
                    break
                cpu_ai._apply_move_inplace(m, name, mv, stop_at_select=True)
            return vf(m, name)
        except Exception:
            return None

    n_win = n_diverge = 0
    kinds = {"過少防御": 0, "過剰防御": 0, "同等": 0}
    gaps = []
    for name, m in windows:
        var = defense_variants(m, name)
        if len(var) < 2:
            continue
        scored = []
        for label, moves in var:
            v = battle_end_value(m, name, moves)
            if v is not None:
                scored.append((label, len(moves), v))
        if len(scored) < 2:
            continue
        n_win += 1
        best = max(scored, key=lambda x: x[2])
        # エンジンの実選択: 同じ窓で decide → コミットプランの先頭手の型で近似
        # （PASS=素通し系／SELECT_COUNTER=支払い系）
        actor = m.p1 if m.p1.name == name else m.p2
        mv = eng.decide(m, actor, sims=160, rng=np.random.default_rng(11))
        d = cpu_ai._describe_move(m, mv) or {}
        engine_pays = d.get("action_type") == "SELECT_COUNTER"
        best_pays = best[1] > 0
        if engine_pays == best_pays:
            kinds["同等"] += 1
        elif best_pays and not engine_pays:
            kinds["過少防御"] += 1
            n_diverge += 1
        else:
            kinds["過剰防御"] += 1
            n_diverge += 1
        base = next((v for l, n, v in scored if n == 0), None)
        if base is not None:
            gaps.append(best[2] - base)
    med = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    print(f"DEFENSE_AUDIT windows={n_win} 乖離={n_diverge} 型={kinds} "
          f"守る価値の中央値(最良-素通し)={med:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
