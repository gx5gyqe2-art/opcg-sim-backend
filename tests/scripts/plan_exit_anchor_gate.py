"""段3裁定アンカーによるターン出口ヘッドの検査（系統1・A段ゲート①・2026-08-20）。

`tests/fixtures/move_audit_stage3.json` の裁定から「正しい手 vs 誤った手」の対を作り、
同一局面から**各手を打ってターンを閉じた出口盤面**の順位を、候補ネットのターン出口評価
（`predict_exit("turn")`＝ヘッド無しなら素の value に等価）が正しく並べるかを数える。

- best_correct の件 → 最良手の出口 > 打った手の出口
- cpu_correct の件 → 打った手の出口 > 最良手の出口
（best_also_wrong / hold は対にしない）

**訓練には使わない**（系統3のみで学習し、ここは完全ホールドアウト＝リークゼロ）。
局面の復元は監査時条件（fixture の conditions）の決定的リプレイ。出口の作り方は
`plan.execute_plan`（serve と同一規約）＝「その1手 → ターンを閉じる」の最小解釈。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python \
    tests/scripts/plan_exit_anchor_gate.py --value opcg_sim/data/learned/gen14_value.npz
  （--value に候補 value.npz を渡して学習前後の正答数を比べる）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import plan as PL

REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
FIXTURE = _os.path.join(REPO, "tests", "fixtures", "move_audit_stage3.json")


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


def anchor_pairs(fixture):
    """裁定 → (loc, 上位に来るべき desc, 下位に来るべき desc) の対（pure）。"""
    out = []
    for it in fixture["items"]:
        v = it.get("verdict")
        if v == "best_correct":
            out.append((it["loc"], "best", "chosen"))
        elif v == "cpu_correct":
            out.append((it["loc"], "chosen", "best"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", required=True, help="検査する value .npz（候補 or 現行）")
    ap.add_argument("--sims", type=int, default=160)
    args = ap.parse_args()

    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core import cpu_learned as CL

    fixture = json.load(open(FIXTURE))
    pairs = anchor_pairs(fixture)
    # regret.jsonl（結果ブランチ）から chosen/best の move desc を引く必要はない —
    # fixture の chosen/best は表示文字列なので、盤面から手を particularize するために
    # 段2の結果行（options 内の move desc）を使う。origin の結果ブランチから読む。
    import subprocess

    def _sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=REPO).stdout

    regrets = {}
    for br in sorted(b for b in _sh(
            "git for-each-ref refs/remotes/origin --format='%(refname:short)'").split()
            if "moveaudit-shard" in b):
        for l in _sh(f"git show {br}:audit_results 2>/dev/null").splitlines():
            if not l.startswith("shard"):
                continue
            for line in _sh(f"git show {br}:audit_results/{l.strip('/')}/regret.jsonl 2>/dev/null").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                regrets.setdefault(f"{r['seed']}@{r['decision']}", r)

    db = _load_db()
    replay_eng = CL.LearnedEngine(sims=args.sims)                     # 復元は監査時条件＝既定ネット
    # 評価側はエンジン不要＝value .npz を直接ロードして出口関数を作る（policy 世代に依存しない）
    import rl_net as RN
    cand_v = RN.ValueNet.load(args.value)
    cand_ev = CL._net_enc_version(cand_v)
    if cand_ev != replay_eng.enc_version:
        # serve と同じ物差しで測る（LearnedEngine は serve 世代へ温スタートする）
        cand_v = CL.warm_start_value(cand_v, cand_ev, replay_eng.enc_version)
        cand_ev = replay_eng.enc_version
    if cand_v.has_exit_head("turn"):
        exit_fn = CL._exit_head_value_fn(cand_v, replay_eng.vocab, cand_ev, kind="turn")
        print(f"評価: turn 出口ヘッドあり（enc v{cand_ev}）", flush=True)
    else:
        exit_fn = CL._value_fn(cand_v, replay_eng.vocab, cand_ev)
        print(f"評価: turn 出口ヘッド無し＝素の value（enc v{cand_ev}）", flush=True)
    vf = CL._value_fn(replay_eng.vnet, replay_eng.vocab, replay_eng.enc_version)
    pf = CL._priors_fn(replay_eng.pnet, replay_eng.vocab, replay_eng.enc_version)

    by_seed = collections.defaultdict(list)
    for loc, hi, lo in pairs:
        seed, dec = loc.split("@")
        by_seed[int(seed)].append((int(dec), loc, hi, lo))

    n_ok = n_all = 0
    details = []
    for seed, rows in sorted(by_seed.items()):
        la, lb = _leader_pair(db, seed, "random")
        cap = _Cap([d for d, _, _, _ in rows])
        seat = make_seat(kind="learned", want_trace=False, sims=args.sims, engine=replay_eng)
        try:
            run_game(seed, db, seats={"p1": seat, "p2": seat},
                     deck_builder=synth_deck_builder(la, lb, seed=seed),
                     observers=(cap,), max_steps=1500, legal_moves="skip",
                     invariants="raise", stop_after_decisions=max(d for d, _, _, _ in rows) + 3)
        except _Done:
            pass
        except BaseException as e:
            print(f"seed {seed}: 再生失敗 {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        for dec, loc, hi, lo in rows:
            got = cap.frames.get(dec)
            r = regrets.get(loc)
            if got is None or r is None:
                print(f"  {loc}: 復元/段2行なし → skip", flush=True)
                continue
            frame, actor = got

            def _exit_of(which):
                desc = r.get(which)
                legal = replay_eng.game.legal_actions(frame)
                mv = next((m for m in legal
                           if (cpu_ai._describe_move(frame, m) or {}) == desc), None)
                if mv is None:
                    return None
                return PL.execute_plan(replay_eng.game, frame.clone(), actor,
                                       [PL.move_sig(mv)], vf, pf)

            e_hi, e_lo = _exit_of(hi), _exit_of(lo)
            if e_hi is None or e_lo is None:
                print(f"  {loc}: 手の同定に失敗 → skip", flush=True)
                continue
            v_hi, v_lo = exit_fn(e_hi, actor), exit_fn(e_lo, actor)
            ok = v_hi > v_lo
            n_ok += int(ok)
            n_all += 1
            details.append((loc, ok, round(v_hi, 4), round(v_lo, 4)))
            print(f"  {loc}: {'OK ' if ok else 'NG '} 上位側 {v_hi:+.4f} vs 下位側 {v_lo:+.4f}",
                  flush=True)

    print(f"\nANCHOR_GATE_RESULT {json.dumps({'value': args.value, 'ok': n_ok, 'total': n_all}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
