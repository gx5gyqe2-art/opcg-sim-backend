"""原因分析プローブ F: プラン読み出し（SERVE_PLAN_READOUT）ON/OFF の介入比較。

ユーザ仮説「根本はターン中のプランが立てられていないこと」の直接検証。
同一ネット（G14・ターン出口ヘッド無し＝出口評価は素ヘッドにフォールバック）で
plan_readout=True のエンジンを問題の局面に置き、
  (1) その1手の選択分布（×5 rng）
  (2) そのターンを最後まで打たせた時のターン線（1本・決定的）
を OFF（本番既定）と比較する。プラン構造だけで「付与の回収漏れ」が消えるかを見る。

出力: /home/user/cause_f.log
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

WANT = {502006: [121, 127, 130], 500003: [86, 90], 503002: [8]}


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
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.core import cpu_ai
    from opcg_game import OPCGGame

    db = _load_db()
    eng_off = LearnedEngine(sims=160)                      # 本番既定（plan OFF）
    eng_on = LearnedEngine(sims=160, plan_readout=True)    # プラン読み出し ON
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
            s += f" {nm(d['card'])}"
        if d.get("targets"):
            s += "→" + ",".join(nm(t) for t in d["targets"])
        return s

    frames = {}
    for seed, decs in WANT.items():
        la, lb = _leader_pair(db, seed, "random")
        cap = _Cap(decs)
        seat = make_seat(kind="learned", want_trace=False, sims=160, engine=eng_off)
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

    def turn_line(frame, actor, eng, max_moves=30):
        """このターンの自席の手を最後まで打たせる（相手判断は本番既定エンジン）。"""
        m = frame.clone()
        eng._world_seeds = {}; eng_off._world_seeds = {}
        rng = np.random.default_rng(11)
        turn0 = int(getattr(m, "turn_count", 0) or 0)
        line = []
        for _ in range(max_moves):
            if m.winner is not None or gs.is_terminal(m):
                break
            name = gs.current_player(m)
            if name is None:
                break
            if int(getattr(m, "turn_count", 0) or 0) != turn0:
                break
            e = eng if name == actor else eng_off
            actor_obj = m.p1 if m.p1.name == name else m.p2
            mv = e.decide(m, actor_obj, rng=rng)
            if mv is None:
                break
            lbl = mvlabel(m, mv)
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                break
            if name == actor:
                line.append(lbl)
            m = m2
            if name == actor and lbl.startswith("TURN_END"):
                break
        return line

    for (seed, dec), (frame, actor) in sorted(frames.items()):
        print("=" * 72, flush=True)
        print(f"■ {seed}@{dec} 席={actor}", flush=True)
        for tag, eng in (("OFF", eng_off), ("ON ", eng_on)):
            cnt = collections.Counter()
            for i in range(5):
                rng = np.random.default_rng(100 + i)
                mv = eng.decide(frame.clone(),
                                frame.p1 if frame.p1.name == actor else frame.p2, rng=rng)
                cnt[mvlabel(frame, mv)] += 1
            print(f"  plan {tag} 1手分布: " + "、".join(f"{k}×{v}" for k, v in cnt.most_common()),
                  flush=True)
        for tag, eng in (("OFF", eng_off), ("ON ", eng_on)):
            line = turn_line(frame, actor, eng)
            print(f"  plan {tag} ターン線: " + " → ".join(line[:14]), flush=True)
    print("CAUSE_F_DONE", flush=True)


if __name__ == "__main__":
    main()
