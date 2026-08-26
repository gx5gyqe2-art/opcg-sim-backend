"""ターン出口教師・系統2＝プラン構造化審判の反実仮想コーパス（B段・2026-08-20）。

v38 `plan_cf_gen` の後継。同じ「同一決定点の候補プランを強制実行→出口を勝率ラベル」だが、
B段の設計（ユーザ議論 2026-08-20）で3点を変える:

  1. **審判を π_plan にする**: ロールアウトは両席とも plan_readout=True のエンジン
     （A段候補 = turn出口ヘッド搭載ネット）。旧審判（1手CPU）は付与を回収できず
     「完遂すれば強い出口」を系統的に過小評価する（原因分析§2＝V^π 汚染）。
  2. **候補に構造化提案を含める**: policy 提案だけでは正解の型が候補に入らない
     （プレイ組×浮ドン・`plan.struct_intents`）。
  3. **世界は決定化（相手手札もサンプル）**: 手札真値固定の世界は情報集合の判断の
     ラベルに不適（原因分析§5.2・#15）。K世界はプラン間で共有（CRN）。

`--drift N` で最初の N 判断点について旧審判（plan OFF・同 sims）でも同じ出口を測り、
**ラベル乖離 |Δz| を報告**する＝「既存流の教師づくりは怪しい」の定量実証。

出力: `plancf2_*.npz`（plancf/plandom と同スキーマ）。学習は
  exit_head_finetune.py --head turn --globs "plandom_*.npz,plancf2_*.npz" ...

実行例（スモーク→コスト実測→規模決定）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/plan_cf2_gen.py \
    --games 2 --seed-base 710000 --workers 2 --drift 4 --out /tmp/plancf2_smoke
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import plan as PL
from opcg_sim.src.learned.config import PLAN_TEMP

ROLLOUT_MAX_STEPS = 600


class _Done(BaseException):
    pass


class _MainCap:
    """各席・各ターンの**最初の**メイン判断の直前 manager を複製（B段の判断点）。"""

    def __init__(self, limit_decisions):
        self.limit = limit_decisions
        self.n = 0
        self.frames = []      # (decision_no, turn, seat, manager)
        self._seen = set()
        self._keys = cpu_ai._pending_keys()

    def on_decision_point(self, ctx):
        m = ctx.manager
        name = getattr(ctx.actor, "name", None)
        _kp, k_action = self._keys
        if (ctx.pending or {}).get(k_action) != "MAIN_ACTION":
            return
        if getattr(getattr(m, "turn_player", None), "name", None) != name:
            return
        turn = int(getattr(m, "turn_count", 0) or 0)
        if (name, turn) in self._seen:
            return
        self._seen.add((name, turn))
        self.frames.append((self.n + 1, turn, name, m.clone()))

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.limit:
            raise _Done()


_G = {}


def _init(cand_spec, sims, rollout_sims, enc_version):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    v, _, p = cand_spec.partition(",")
    # 局面採取は本番既定＝出荷分布。**注**: プラン読み出しの serve 配線は純正AZ化
    # （2026-08-25）で削除されたため、旧 π_plan 審判（plan_readout=True）は組めない＝
    # ref_plan/ref_old は同一構成（候補ネット＋標準 decide）になる。--drift の乖離測定は
    # 実質無効（計器としての立案系 API＝plan.py はそのまま使う）。
    _G["base"] = CL.LearnedEngine(sims=sims)
    _G["ref_plan"] = CL.LearnedEngine(value_path=v or None, policy_path=p or None,
                                      sims=rollout_sims)
    _G["ref_old"] = CL.LearnedEngine(value_path=v or None, policy_path=p or None,
                                     sims=rollout_sims)
    eng = _G["base"]
    _G["vf"] = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    _G["pf"] = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    _G["enc_version"] = enc_version or eng.enc_version
    # 審判の立案を軽量化: ロールアウト内の select_plan は世界3・policy提案3で足りる
    # （ラベルは順位ペア化するので審判の精度は本番程でなくて良い＝コストが支配的）。
    # n_worlds/n_proposals は def 時に束縛された既定引数のためラッパで上書きする。
    _orig_select = PL.select_plan
    def _cheap_select(game, manager, name, value_fn, priors_fn, rng, **kw):
        kw.setdefault("n_worlds", 3)
        kw.setdefault("n_proposals", 3)
        return _orig_select(game, manager, name, value_fn, priors_fn, rng, **kw)
    PL.select_plan = _cheap_select
    PL.PLAN_STRUCT_MAX = 4


def candidate_plans(frame, name, rng, n_policy=3, n_struct=4, cap=6):
    """候補プラン（policy argmax/温度＋構造化）を signature 列で返す（重複除去・上限 cap）。"""
    eng, vf, pf = _G["base"], _G["vf"], _G["pf"]
    plans, labels = [], []

    def _add(steps, lab):
        if steps and steps not in plans and len(plans) < cap:
            plans.append(list(steps))
            labels.append(lab)

    for k in range(n_policy):
        w = eng.game.determinize(frame, name, rng)
        t = 0.0 if k == 0 else PLAN_TEMP
        _add(PL.rollout_plan(eng.game, w, name, vf, pf, rng, temp=t),
             "policy:argmax" if k == 0 else "policy:temp")
    for lab, intent in PL.struct_intents(frame, name)[:n_struct]:
        w = eng.game.determinize(frame, name, rng)
        _add(PL.scripted_plan(eng.game, w, name, intent, vf, pf), lab)
    return plans, labels


def _rollout_winner(gs, m, ref_eng, rng_seed, max_turns=0):
    """出口から両席 ref_eng（π_plan または旧π）で打つ。返り値 (winner, 最終盤面)。

    `max_turns`>0 なら**そのターン数だけ進めて打ち切る**（TD式短縮）。終局に届かない場合は
    呼び出し側が最終盤面を value でブートストラップしてラベル化する＝フル終局ロールアウトが
    高すぎる場合のコストつまみ（順位ペア化するので粗いラベルで足りる）。"""
    rng = np.random.default_rng(rng_seed)
    steps = 0
    turn0 = int(getattr(m, "turn_count", 0) or 0)
    while m.winner is None and not gs.is_terminal(m) and steps < ROLLOUT_MAX_STEPS:
        if max_turns and int(getattr(m, "turn_count", 0) or 0) - turn0 >= max_turns:
            break
        name = gs.current_player(m)
        if name is None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        mv = ref_eng.decide(m, actor, rng=rng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            break
        m = m2
        steps += 1
    return m.winner, m


def label_plans(frame, name, plans, worlds, ref_key, max_turns=0):
    """各プランを K 決定化世界（プラン間共有＝CRN）で出口→終局（または M ターン打ち切り＋
    value ブートストラップ）し、z の平均を返す。"""
    eng, vf, pf = _G["base"], _G["vf"], _G["pf"]
    gs = eng.game
    ref = _G[ref_key]
    zs = []
    for steps in plans:
        acc, n = 0.0, 0
        for wi, world in enumerate(worlds):
            exit_mgr = PL.execute_plan(gs, world, name, list(steps), vf, pf)
            ref._world_seeds = {}
            ref._turn_plans = {}
            w, last = _rollout_winner(gs, exit_mgr, ref, 91000 + wi * 13 + 1,
                                      max_turns=max_turns)
            if w is not None:
                acc += 1.0 if w == name else -1.0
                n += 1
            elif max_turns and last is not None:
                acc += float(_G["vf"](last, name))     # 打ち切り＝value ブートストラップ
                n += 1
        zs.append(acc / n if n else None)
    return zs


def _run_game(job):
    seed, args = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.learned import encoder as E
    db, eng = _G["db"], _G["base"]
    la, lb = _leader_pair(db, seed, "random")
    cap = _MainCap(limit_decisions=200)
    seat = make_seat(kind="learned", want_trace=False, sims=args["sims"], engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=200)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}",
                "rows": [], "drift": []}

    # 判断点は中盤帯（T3〜14）から等間隔に採る。序盤は自明・終盤は決着済みで
    # 打ち切りロールアウトが全プラン ±1 に飽和し順位を教えられない（スモーク実測）。
    frames = [f for f in cap.frames if 3 <= f[1] <= 14] or cap.frames
    take = args["points"]
    if len(frames) > take:
        idx = np.linspace(0, len(frames) - 1, take).astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]

    rows, drifts = [], []
    gi = 0
    for dec, turn, name, frame in frames:
        t0 = time.time()
        rng = np.random.default_rng(seed * 7 + dec)
        plans, labels = candidate_plans(frame, name, rng)
        if len(plans) < 2:
            continue
        worlds = []
        for wi in range(args["worlds"]):
            try:
                worlds.append(eng.game.determinize(frame, name,
                                                   np.random.default_rng(88000 + wi)))
            except Exception:
                break
        if len(worlds) < 2:
            continue
        zs = label_plans(frame, name, plans, worlds, "ref_plan", args["rollout_turns"])
        if args["drift_left"] > 0:
            zo = label_plans(frame, name, plans, worlds, "ref_old", args["rollout_turns"])
            ds = [abs(a - b) for a, b in zip(zs, zo) if a is not None and b is not None]
            if ds:
                drifts.append(float(np.mean(ds)))
            args["drift_left"] -= 1
        ok = [(p, z) for p, z in zip(plans, zs) if z is not None]
        spread = (max(z for _, z in ok) - min(z for _, z in ok)) if len(ok) >= 2 else 0.0
        if len(ok) < 2 or spread < 0.05:
            print(f"  seed {seed}@{dec} T{turn}: skip（有効{len(ok)}本・幅{spread:.3f}） "
                  f"{time.time() - t0:.0f}s", flush=True)
            continue          # 全滅/飽和の決定点は捨てる（順位を教えられない）
        for steps, z in ok:
            exit_mgr = PL.execute_plan(eng.game, frame.clone(), name, list(steps),
                                       _G["vf"], _G["pf"])
            enc = E.encode(exit_mgr, name, eng.vocab, version=_G["enc_version"])
            rows.append((enc, float(z), gi))
        gi += 1
        print(f"  seed {seed}@{dec} T{turn}: プラン{len(plans)}本 z範囲 "
              f"[{min(z for _, z in ok):+.2f},{max(z for _, z in ok):+.2f}] "
              f"{time.time() - t0:.0f}s", flush=True)
    return {"seed": seed, "error": None, "rows": rows, "drift": drifts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=710000)
    ap.add_argument("--points", type=int, default=2, help="1局あたりの判断点数（ターン全域から等間隔）")
    ap.add_argument("--worlds", type=int, default=4, help="決定化世界の数（プラン間共有＝CRN）")
    ap.add_argument("--sims", type=int, default=160, help="局面採取の自己対戦（分布=本番仕様）")
    ap.add_argument("--rollout-turns", type=int, default=0,
                    help="ロールアウトを M ターンで打ち切り value でブートストラップ（0=終局まで）")
    ap.add_argument("--rollout-sims", type=int, default=40,
                    help="審判ロールアウトの sims（教師は順位ペア化するので低くて良い）")
    ap.add_argument("--cand", default="", help="審判ネット value,policy（空=既定G14）")
    ap.add_argument("--drift", type=int, default=0,
                    help="最初の N 判断点で旧審判（plan OFF）ともラベルし |Δz| を測る")
    ap.add_argument("--enc-version", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.seed_base + i,
             {"sims": args.sims, "points": args.points, "worlds": args.worlds,
              "drift_left": 1 if i < args.drift else 0,
              "rollout_turns": args.rollout_turns}) for i in range(args.games)]
    buf, stats, drifts = [], {"rows": 0, "groups": 0, "errors": 0}, []
    shard = [0]

    def _flush(chunk):
        if not chunk:
            return
        arrays = {
            "scalars": np.stack([r[0]["scalars"] for r in chunk]).astype(np.float32),
            "field": np.stack([r[0]["field"] for r in chunk]).astype(np.float32),
            "card_idx": np.stack([r[0]["card_idx"] for r in chunk]).astype(np.int32),
            "value": np.array([r[1] for r in chunk], dtype=np.float32),
            "group": np.array([r[2] for r in chunk], dtype=np.int64),
        }
        np.savez_compressed(os.path.join(args.out, f"plancf2_{shard[0]:05d}.npz"), **arrays)
        shard[0] += 1

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.cand, args.sims, args.rollout_sims, args.enc_version)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                stats["errors"] += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            gbase = res["seed"] * 1000
            gseen = set()
            for enc, z, g in res["rows"]:
                buf.append((enc, z, gbase + g))
                gseen.add(g)
            stats["rows"] += len(res["rows"])
            stats["groups"] += len(gseen)
            drifts += res["drift"]
            while len(buf) >= args.shard_size:
                _flush(buf[:args.shard_size])
                buf = buf[args.shard_size:]
    _flush(buf)
    dmsg = (f" drift|Δz|={np.mean(drifts):.3f}(n={len(drifts)})" if drifts else "")
    print(f"PLAN_CF2_DONE rows={stats['rows']} groups={stats['groups']} "
          f"errors={stats['errors']} shards={shard[0]}{dmsg} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
