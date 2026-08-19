"""原因分析プローブ A/B/C（2026-08-19・段3裁定 #1,2,5,13,14 の機序特定）。

A: 各選択肢を適用した後状態を価値ネットに直接評価させる（V(s') の識別力）
B: sims=40/160/640 × 5反復の decide（探索量で誤りが消えるか）
C: 1回の traced decide でルートの P/Q/N 上位を出す（プライア vs 読み出し）

出力: /home/user/cause_abc.log（人間可読）
"""
import os, sys, json, collections, subprocess
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

WANT = {502006: [121, 127, 130], 500003: [86, 90], 506006: [101], 500000: [31]}


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
    from opcg_sim.src.core import cpu_ai
    from opcg_game import OPCGGame

    db = _load_db()
    eng = CL.LearnedEngine(sims=160)
    vfn = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version, aux_tiebreak=False)
    gr = OPCGGame(prune_futile=False)
    gs = OPCGGame()

    def nm(cid):
        try:
            return db.get_card(cid).name
        except Exception:
            return cid

    def mvlabel(m, mv):
        d = cpu_ai._describe_move(m, mv) or {}
        s = d.get("action_type", "?")
        if d.get("card"):
            s += f" {nm(d['card'])}({d['card']})"
        if d.get("targets"):
            s += "→" + ",".join(nm(t) for t in d["targets"])
        return s, d

    frames = {}
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
        except BaseException as e:
            print(f"seed {seed}: 再生失敗 {type(e).__name__}: {str(e)[:80]}", flush=True)
        for d, fr in cap.frames.items():
            frames[(seed, d)] = fr
        print(f"seed {seed}: frames {sorted(cap.frames)}", flush=True)

    for (seed, dec), (frame, actor) in sorted(frames.items()):
        print("=" * 72, flush=True)
        print(f"■ {seed}@{dec} 席={actor}", flush=True)
        legal = gr.legal_actions(frame)
        # 等価マージ
        uniq, seen = [], set()
        for mv in legal:
            k = cpu_ai._move_equiv_key(frame, mv)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(mv)
        # --- A: V(s') 直接評価 ---
        print("-- A: 各選択肢の後状態 V(s')（席視点・探索なし）", flush=True)
        rows = []
        for mv in uniq:
            lbl, d = mvlabel(frame, mv)
            m2 = gs.apply(frame.clone(), mv, actor)
            if m2 is None:
                continue
            rows.append((float(vfn(m2, actor)), lbl))
        for v, lbl in sorted(rows, reverse=True):
            print(f"   V'={v:+.3f}  {lbl}", flush=True)
        # --- C: traced decide（P/Q/N 上位） ---
        print("-- C: traced decide (sims=160)", flush=True)
        tr = {}
        rng = np.random.default_rng(7)
        mv = eng.decide(frame.clone(), frame.p1 if frame.p1.name == actor else frame.p2,
                        rng=rng, trace=tr)
        lbl, _ = mvlabel(frame, mv)
        print(f"   chosen: {lbl}  readout={tr.get('readout')}", flush=True)
        print(f"   q_margin={tr.get('q_margin')} policy_rank={tr.get('policy_rank')} "
              f"policy_top={tr.get('policy_top')}", flush=True)
        # --- B: sims スケーリング ---
        print("-- B: sims=40/160/640 ×5 の decide 分布", flush=True)
        for sims in (40, 160, 640):
            cnt = collections.Counter()
            e2 = CL.LearnedEngine(sims=sims)
            for i in range(5):
                rng = np.random.default_rng(100 + i)
                mv = e2.decide(frame.clone(),
                               frame.p1 if frame.p1.name == actor else frame.p2, rng=rng)
                lbl, _ = mvlabel(frame, mv)
                cnt[lbl] += 1
            top = "、".join(f"{k}×{v}" for k, v in cnt.most_common())
            print(f"   sims={sims:>3}: {top}", flush=True)
    print("CAUSE_ABC_DONE", flush=True)


if __name__ == "__main__":
    main()
