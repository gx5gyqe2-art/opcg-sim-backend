"""リーサル帯・乖離盤面の教師採掘（G系 v51・2026-08-12・`lethal_calibration_probe` の対）。

**目的**: v50 で確定した G14 の欠陥族「見かけと実質が乖離した盤面」（ライフ帯で value が
確信を持って間違える・最悪誤差 −1.88）の較正教師を、**盤面ごとの証明**で選別して量産する。

**採掘規則（ユーザ合意 2026-08-12）**:
  ① 非エネル席のみ（既定対面 nami:shanks＝v47 実測で較正健全・訓練分布の中心）
  ② ラベルは**実現による自己証明**があるもののみ: |EV| ≥ 2/3（6世界中5世界以上を勝った側が
     実演した＝「対面への信頼」でなく「この盤面でこの結末が実現できた」という構成的証拠）
  ③ 教師化するのは value が**確信を持って**（|予測| ≥ 0.5）**外している**
     （|予測 − EV| ≥ pred-gap 既定0.5）盤面だけ＝乖離族に的を絞り一般盤面は動かさない
  ④ ラベル器は教師正本（CR.rollout sims48 def_temp0.7）＋真値世界（山札シャッフルのみ CRN）

出力: G系の通常符号化（v9・card_idx 実ID込み）＋ value=EV の npz 行
（`dense_finetune` 系の MSE 微調整が読む形・q_root=NaN 勝敗単独）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/lethal_teacher_gen.py \\
    --games 120 --workers 3 --out /tmp/v51_teacher
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
MAX_STEPS = 400
_G = {}


def _init_worker(matchup, gen_sims, label_sims):
    import counterfactual_referee as CR
    import p3_loop as P
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn
    a, b = matchup.split(":")
    CR.ARGS = argparse.Namespace(sims=label_sims, true_board=False)
    db = _load_db()
    eng = LearnedEngine()
    _G.update(CR=CR, E=E, db=db, eng=eng,
              game_gen=_make_fixed_matchup_game(DECKS_JSON, a, b),
              gs=OPCGGame(), gen_sims=gen_sims,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
              vpred=_value_fn(eng.vnet, eng.vocab, eng.enc_version))


def _shuffle_decks(m, w):
    for pid_i, pl in enumerate((m.p1, m.p2)):
        r = np.random.default_rng(70000 + w * 101 + pid_i)
        order = r.permutation(len(pl.deck))
        pl.deck[:] = [pl.deck[int(i)] for i in order]


def play_one(task):
    """1局: 自己対戦→リーサル帯スナップショット→確信上位を教師正本でラベル→証明選別。"""
    seed, cfg = task
    CR, E, gs, eng = _G["CR"], _G["E"], _G["gs"], _G["eng"]
    m = _G["game_gen"].new_game(_G["db"], seed)
    drng = np.random.default_rng(seed * 17 + 3)
    snaps, seen, steps = [], set(), 0
    while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
        name = gs.current_player(m)
        if name is None:
            break
        me = m.p1 if m.p1.name == name else m.p2
        opp = m.p2 if m.p1.name == name else m.p1
        t = int(getattr(m, "turn_count", 0) or 0)
        key = (t, name)
        if (key not in seen and t >= cfg["min_turn"]
                and min(len(me.life or []), len(opp.life or [])) <= cfg["max_life"]):
            seen.add(key)
            snaps.append((m, name, t))         # apply はクローンを返す＝参照保持で凍結
        actor = m.p1 if m.p1.name == name else m.p2
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=_G["gen_sims"], rng=drng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            break
        m = m2
        steps += 1

    # 確信の強い盤面から最大 boards_per_game 点をラベル化
    scored = []
    for m0, name, t in snaps:
        pred = _G["vpred"](m0, name)
        if abs(pred) >= cfg["pred_min"]:
            scored.append((abs(pred), pred, m0, name, t))
    scored.sort(key=lambda x: -x[0])
    rows, diag = [], []
    for _apred, pred, m0, name, t in scored[:cfg["boards_per_game"]]:
        wins = ok = 0
        for w in range(cfg["worlds"]):
            mw = m0.clone()
            _shuffle_decks(mw, w)
            try:
                wn, _ld, _et = CR.rollout(gs, _G["vf"], _G["pf"], mw, name,
                                          world_seed=71000 + w, rng_seed=(71000 + w) * 131,
                                          def_temp=0.7)
            except Exception:
                continue
            ok += 1
            wins += 1 if wn == name else 0
        if ok == 0:
            continue
        ev = 2.0 * wins / ok - 1.0
        certified = abs(ev) >= 2.0 / 3.0       # 実現による自己証明（勝った側が5/6以上実演）
        gap = pred - ev
        diag.append({"seed": seed, "turn": t, "who": name, "pred": round(pred, 3),
                     "wr": f"{wins}/{ok}", "ev": round(ev, 3),
                     "cert": certified, "teach": bool(certified and abs(gap) >= cfg["pred_gap"])})
        if certified and abs(gap) >= cfg["pred_gap"]:
            enc = E.encode(m0, name, eng.vocab, version=eng.enc_version)
            rows.append((enc["scalars"], enc["field"], enc["card_idx"], ev))
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gen-sims", type=int, default=32)
    ap.add_argument("--label-sims", type=int, default=48, help="教師正本（CR canon）")
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--min-turn", type=int, default=5)
    ap.add_argument("--max-life", type=int, default=1, help="min(両者ライフ)≤この値の盤面を採る")
    ap.add_argument("--pred-min", type=float, default=0.5, help="③ 確信の下限")
    ap.add_argument("--pred-gap", type=float, default=0.5, help="③ 乖離の下限")
    ap.add_argument("--boards-per-game", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = {k: getattr(args, k.replace("-", "_")) for k in
           ("worlds", "min_turn", "max_life", "pred_min", "pred_gap", "boards_per_game")}
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    all_rows, all_diag = [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.matchup, args.gen_sims,
                                                args.label_sims)) as pool:
        done = 0
        for rows, diag in pool.imap_unordered(
                play_one, [(args.seed_base + i, cfg) for i in range(args.games)]):
            done += 1
            all_rows += rows
            all_diag += diag
            n_teach = sum(1 for d in all_diag if d["teach"])
            if done % 10 == 0 or rows:
                print(f"  {done}/{args.games}局 ラベル{len(all_diag)} 教師{n_teach} "
                      f"{time.time()-t0:.0f}s", flush=True)
    if all_rows:
        np.savez_compressed(
            os.path.join(args.out, "v51t_00000.npz"),
            scalars=np.array([r[0] for r in all_rows], np.float32),
            field=np.array([r[1] for r in all_rows], np.float32),
            card_idx=np.array([r[2] for r in all_rows], np.int64),
            value=np.array([r[3] for r in all_rows], np.float32),
            q_root=np.full(len(all_rows), np.nan, np.float32),
            turns_left=np.full(len(all_rows), np.nan, np.float32))
    with open(os.path.join(args.out, "meta_v51t.json"), "w") as f:
        json.dump({"games": args.games, "labeled": len(all_diag),
                   "teachers": len(all_rows), "cfg": cfg, "matchup": args.matchup,
                   "diag": all_diag}, f, ensure_ascii=False, indent=1)
    print("V51_TEACHER_DONE " + json.dumps(
        {"games": args.games, "labeled": len(all_diag), "teachers": len(all_rows)},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
