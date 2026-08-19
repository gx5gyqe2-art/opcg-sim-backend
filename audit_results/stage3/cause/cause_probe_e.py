"""原因分析プローブ E（#15・P4検証）: 501001@95 でユーザ指定のリーサル台本を強制測定。

段2の測定は「打った手 wr=0.000／最良（ミノチワワ登場）wr=0.333」だったが、ユーザ裁定は
「どちらも誤り＝ドンを全部付与してリーダーへ総攻撃が正解」。審判（現行CPU）はこの手順を
自分では生成できない。そこで**攻撃側のこのターンだけ台本**（ドンを攻撃者3体へ分配 →
高パワー順に相手リーダーへ総攻撃 → ターン終了）で強制し、守備側と以降の進行は本番CPUの
まま同一世界CRNで勝率を測る。台本の勝率が 0.333 を大きく上回れば「審判の手生成の限界」
が確定する。

出力: /home/user/cause_e.log
"""
import os, sys, collections
os.environ.setdefault("OPCG_LOG_SILENT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, f"{REPO}/tests")
sys.path.insert(0, f"{REPO}/tests/scripts")
sys.path.insert(0, f"{REPO}/tests/harness")
os.chdir(REPO)
import _bootstrap  # noqa
import numpy as np

SEED, DEC = 501001, 95
WORLDS = 12
ROLLOUT_MAX_STEPS = 600


class _Done(BaseException):
    pass


class _Cap:
    def __init__(self, want):
        self.want = want; self.n = 0; self.frame = None

    def on_decision_point(self, ctx):
        if self.n + 1 == self.want:
            self.frame = (ctx.manager.clone(), ctx.actor.name)

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n >= self.want:
            raise _Done()


def main():
    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.core import cpu_ai
    from opcg_game import OPCGGame

    db = _load_db()
    eng = LearnedEngine(sims=160)
    gr = OPCGGame(prune_futile=False)
    gs = OPCGGame()

    la, lb = _leader_pair(db, SEED, "random")
    cap = _Cap(DEC)
    seat = make_seat(kind="learned", want_trace=False, sims=160, engine=eng)
    try:
        run_game(SEED, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=SEED),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=DEC + 2)
    except _Done:
        pass
    frame, me = cap.frame
    print(f"局面復元 {SEED}@{DEC} 席={me}", flush=True)

    def _shuffle(m, w):
        for i, pl in enumerate((m.p1, m.p2)):
            r = np.random.default_rng(90000 + w * 101 + i)
            order = r.permutation(len(pl.deck))
            pl.deck[:] = [pl.deck[int(i2)] for i2 in order]

    def scripted_move(m, name, turn0):
        """台本: 自ターンの MAIN で「均等にドン分配 → 高パワー順にリーダー総攻撃 → 終了」。
        戦闘窓の自分側は PASS。その他（効果ダイアログ等）は None を返し本番 decide に委ねる。"""
        legal = gr.legal_actions(m)
        if not legal:
            return None
        pmap = {}
        meP = m.p1 if m.p1.name == name else m.p2
        opp = m.p2 if m.p1.name == name else m.p1
        for u in ([meP.leader] if meP.leader is not None else []) + list(meP.field):
            pmap[u.uuid] = u
        lid = opp.leader.uuid if opp.leader is not None else None
        attach = [mv for mv in legal if mv.get("action_type") == "ATTACH_DON"
                  and (mv.get("payload") or {}).get("uuid") in pmap
                  and not getattr(pmap[(mv.get("payload") or {}).get("uuid")], "is_rest", False)]
        attacks = [mv for mv in legal if mv.get("action_type") == "ATTACK"
                   and ((mv.get("payload") or {}).get("target_ids") or [None])[0] == lid]
        if attach:
            # 付与: アクティブ攻撃者のうち付与ドンが最少の相手へ（均等分配）
            def dn(mv):
                c = pmap[(mv.get("payload") or {}).get("uuid")]
                ad = getattr(c, "attached_don", None)
                return ad if isinstance(ad, int) else len(ad or [])
            return min(attach, key=dn)
        if attacks:
            def pw(mv):
                c = pmap[(mv.get("payload") or {}).get("uuid")]
                try:
                    return float(c.get_power(True))
                except Exception:
                    return float(getattr(getattr(c, "master", None), "power", 0) or 0)
            return max(attacks, key=pw)
        te = [mv for mv in legal if mv.get("action_type") == "TURN_END"]
        if te:
            return te[0]
        ps = [mv for mv in legal if mv.get("action_type") == "PASS"]
        if ps:
            return ps[0]
        return None

    def play_world(w, scripted):
        m = frame.clone()
        _shuffle(m, w)
        eng._world_seeds = {}
        rng = np.random.default_rng(90000 + w * 7 + 1)
        turn0 = int(getattr(m, "turn_count", 0) or 0)
        steps = 0
        in_script = scripted
        while m.winner is None and not gs.is_terminal(m) and steps < ROLLOUT_MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
            mv = None
            if in_script and name == me:
                if int(getattr(m, "turn_count", 0) or 0) != turn0:
                    in_script = False    # 台本はこのターン限り
                else:
                    mv = scripted_move(m, name, turn0)
            if mv is None:
                actor = m.p1 if m.p1.name == name else m.p2
                mv = eng.decide(m, actor, rng=rng)
            if mv is None:
                break
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                break
            m, steps = m2, steps + 1
        return 1.0 if m.winner == me else 0.0

    for label, scripted in (("台本（ドン分配→リーダー総攻撃）", True),
                            ("本番CPUそのまま（参照）", False)):
        wins6 = sum(play_world(w, scripted) for w in range(6))
        wins12 = wins6 + sum(play_world(w, scripted) for w in range(6, WORLDS))
        print(f"■ {label}: wr(worlds0-5)={wins6/6:.3f}  wr(worlds0-11)={wins12/12:.3f}",
              flush=True)
    print("段2の測定（worlds=6）: 打った手(ビスタ攻撃)=0.000 / 最良(ミノチワワ登場)=0.333",
          flush=True)
    print("CAUSE_E_DONE", flush=True)


if __name__ == "__main__":
    main()
