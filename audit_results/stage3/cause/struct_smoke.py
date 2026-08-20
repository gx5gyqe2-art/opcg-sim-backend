"""構造化提案器スモーク: 段3の問題局面で候補と選択がどう変わるか（2026-08-20）。

502006@121/@127/@130（ドン浪費ターン）で select_plan を直接呼び、
  - 候補プラン（kinds ラベル・スコア）
  - 選ばれたプラン（先頭数手）
を PLAN_STRUCT_PROPOSALS の ON/OFF で比較する。評価は素ヘッド（turnヘッド無し）のまま＝
提案器が「正解の型を候補に入れるか」を見るのが目的（選択の正しさはヘッド訓練後の話）。
"""
import os, sys
os.environ.setdefault("OPCG_LOG_SILENT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, f"{REPO}/tests")
sys.path.insert(0, f"{REPO}/tests/harness")
os.chdir(REPO)
import _bootstrap  # noqa
import numpy as np

WANT = {502006: [121, 127, 130]}


class _Done(BaseException):
    pass


class _Cap:
    def __init__(self, wanted):
        self.wanted = set(wanted); self.n = 0; self.frames = {}
        self.last = max(wanted)

    def on_decision_point(self, ctx):
        if (self.n + 1) in self.wanted:
            self.frames[self.n + 1] = (ctx.manager.clone(), ctx.actor.name)

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.last:
            raise _Done()


def main():
    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core import cpu_learned as CL
    from opcg_sim.src.learned import plan as PL, config as CFG

    db = _load_db()
    eng = CL.LearnedEngine(sims=160)

    def nm(u, mgr):
        for p in (mgr.p1, mgr.p2):
            for c in ([p.leader] if p.leader else []) + list(p.field) + list(p.hand) \
                    + ([p.stage] if p.stage else []):
                if c is not None and getattr(c, "uuid", None) == u:
                    return getattr(c.master, "name", u)
        return (u or "?")[:6]

    def show(steps, mgr, k=8):
        out = []
        for s in steps[:k]:
            at, u = s[0], s[1]
            out.append(f"{at} {nm(u, mgr)}" if u else at)
        return " → ".join(out) + ("…" if len(steps) > k else "")

    for seed, decs in WANT.items():
        la, lb = _leader_pair(db, seed, "random")
        cap = _Cap(decs)
        seat = make_seat(kind="learned", want_trace=False, sims=160, engine=eng)
        try:
            run_game(seed, db, seats={"p1": seat, "p2": seat},
                     deck_builder=synth_deck_builder(la, lb, seed=seed),
                     observers=(cap,), max_steps=1500, legal_moves="skip",
                     invariants="raise", stop_after_decisions=max(decs) + 5)
        except _Done:
            pass
        for d in decs:
            got = cap.frames.get(d)
            if not got:
                print(f"■ {seed}@{d}: 復元失敗", flush=True)
                continue
            frame, actor = got
            vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
            pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
            print("=" * 70, flush=True)
            print(f"■ {seed}@{d} 席={actor}", flush=True)
            for flag in (False, True):
                CFG.PLAN_STRUCT_PROPOSALS = flag
                PL.PLAN_STRUCT_PROPOSALS = flag
                steps, diag = PL.select_plan(eng.game, frame, actor, vf, pf,
                                             np.random.default_rng(7),
                                             exit_value_fn=None, min_spread=0.0)
                tag = "struct ON " if flag else "struct OFF"
                if steps is None:
                    print(f"  [{tag}] 候補なし {diag}", flush=True)
                    continue
                kinds = diag.get("kinds") or []
                scores = diag.get("scores") or []
                print(f"  [{tag}] 候補{diag['n_plans']}本:", flush=True)
                for kk, ss in sorted(zip(kinds, scores), key=lambda x: -x[1]):
                    print(f"      {ss:+.3f}  {kk}", flush=True)
                print(f"    選択: {kinds[diag['best']]}", flush=True)
                print(f"    列  : {show(steps, frame)}", flush=True)
    print("STRUCT_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
