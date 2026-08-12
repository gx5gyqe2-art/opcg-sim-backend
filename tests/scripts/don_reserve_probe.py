"""温存診断プローブ（ドン箱化 P2-a・`docs/cpu_don_box_plan.md` §1 基本3「イベントカウンター用の
相手ターンへの温存」・2026-08-12）。

**問い**: コスト付きカウンターイベントを握る側の「最後のドンを使い切るか残すか」の決定で、
G14 は温存を選べているか。ユーザの同通貨理論（2026-08-12）: 温存ドンのオプション価値＝
「来ターンに使えるカウンター値」＝手札の印字カウンターと同じ防御予算。

**測定点**: 固定対面（既定 bg_luffy:nami・bg_luffy 側は cost1 カウンターイベント×8）の
自己対戦で、(a) 手札にコスト付きカウンターイベント (b) 残ドンがそのコスト境界帯
（min_cost ≤ don ≤ min_cost+1） (c) メイン判断、の点を採る。

**判定**: CPU の選択（sims=160 本番）を分類し、**ドンをしきい値未満へ落とす手を選んだ点**で
2線を教師正本（CR.rollout sims48×6世界 CRN）でラベルする:
  EV_spend = CPU の選択を適用した状態の EV
  EV_end   = 同じ点で即 TURN_END（最も怠惰な温存線＝温存側の下界）の EV
EV_end − EV_spend ≥ 1/3 なら**欠陥点**（最も怠惰な温存にも負ける消費）。逆は「不明」
（より賢い温存線が勝つ可能性は残る＝過小検出側に倒す）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/don_reserve_probe.py \\
    --games 12 --workers 3
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

REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DECKS_JSON = _os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
MAX_STEPS = 400

_G = {}


def _init_worker(matchup, label_sims, serve_sims):
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    a, b = matchup.split(":")
    CR.ARGS = argparse.Namespace(sims=label_sims, true_board=False)
    db = _load_db()
    eng = LearnedEngine()
    _G.update(CR=CR, db=db, eng=eng, probe_side_deck=a,
              game_gen=_make_fixed_matchup_game(DECKS_JSON, a, b),
              gs=OPCGGame(), serve_sims=serve_sims,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version))


def _costed_counter_events(pl):
    """手札のコスト付きカウンターイベント（温存が意味を持つ札）。"""
    from opcg_sim.src.models.enums import CardType, TriggerType
    out = []
    for c in pl.hand:
        m = c.master
        if m.type == CardType.EVENT and (m.cost or 0) > 0 \
                and any(ab.trigger == TriggerType.COUNTER for ab in (m.abilities or [])):
            out.append(c)
    return out


def _ev(m0, name):
    """教師正本 EV（CR.rollout sims=ARGS.sims × 6世界 CRN・teacher_gen と同一規約）。"""
    CR, gs = _G["CR"], _G["gs"]
    wins = ok = 0
    for w in range(6):
        mw = m0.clone()
        for pid_i, pl in enumerate((mw.p1, mw.p2)):
            r = np.random.default_rng(70000 + w * 101 + pid_i)
            order = r.permutation(len(pl.deck))
            pl.deck[:] = [pl.deck[int(i)] for i in order]
        try:
            wn, _ld, _et = CR.rollout(gs, _G["vf"], _G["pf"], mw, name,
                                      world_seed=71000 + w, rng_seed=(71000 + w) * 131,
                                      def_temp=0.7)
        except Exception:
            continue
        ok += 1
        wins += 1 if wn == name else 0
    return (2.0 * wins / ok - 1.0, f"{wins}/{ok}") if ok else (None, "0/0")


def play_one(seed):
    """1局: 自己対戦を進めつつ境界帯のメイン判断を検査（最大2点/局）。"""
    gs, eng = _G["gs"], _G["eng"]
    m = _G["game_gen"].new_game(_G["db"], seed)
    # 検査対象席: probe 側デッキ（seed 偶奇で席が入れ替わる＝リーダー card_id で判定）
    import json as _json
    probe_leader = _json.load(open(DECKS_JSON))[_G["probe_side_deck"]]["leader"]
    probe_name = m.p1.name if m.p1.leader.master.card_id == probe_leader else m.p2.name
    drng = np.random.default_rng(seed * 17 + 3)
    rows, steps, seen = [], 0, set()
    while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS and len(rows) < 2:
        name = gs.current_player(m)
        if name is None:
            break
        t = int(getattr(m, "turn_count", 0) or 0)
        me = m.p1 if m.p1.name == name else m.p2
        evs = _costed_counter_events(me) if name == probe_name else []
        min_cost = min((int(c.master.cost or 0) for c in evs), default=0)
        don = len(me.don_active)
        key = (t, name)
        if (evs and t >= 5 and key not in seen and m.active_battle is None
                and min_cost <= don <= min_cost + 1
                and getattr(m, "turn_player", None) is me):
            seen.add(key)
            eng._world_seeds = {}
            mv = eng.decide(m, me, sims=_G["serve_sims"], rng=np.random.default_rng(seed * 7 + t))
            if mv is None:
                break
            m_after = gs.apply(m, mv, name)
            if m_after is None:
                break
            me_after = m_after.p1 if m_after.p1.name == name else m_after.p2
            crossed = len(me_after.don_active) < min_cost
            at = mv.get("action_type")
            row = {"seed": seed, "turn": t, "who": name, "don": don, "min_cost": min_cost,
                   "n_events": len(evs), "choice": at, "crossed": bool(crossed)}
            if crossed:
                # 欠陥判定: 消費線 vs 最も怠惰な温存線（即 TURN_END）
                ev_spend, wr_s = _ev(m_after, name)
                m_end = gs.apply(m, {"kind": "game", "action_type": "TURN_END", "payload": {}},
                                 name)
                ev_end, wr_e = (None, "-") if m_end is None else _ev(m_end, name)
                if ev_spend is not None and ev_end is not None:
                    row.update({"ev_spend": round(ev_spend, 3), "wr_spend": wr_s,
                                "ev_end": round(ev_end, 3), "wr_end": wr_e,
                                "gap": round(ev_end - ev_spend, 3),
                                "defect": bool(ev_end - ev_spend >= 1.0 / 3.0)})
            rows.append(row)
            m = m_after
            steps += 1
            continue
        actor = m.p1 if m.p1.name == name else m.p2
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=32, rng=drng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            break
        m = m2
        steps += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", default="bg_luffy:nami",
                    help="'守り側デッキ:相手'（守り側がコスト付きカウンターイベント保持側）")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=97000)
    ap.add_argument("--label-sims", type=int, default=48, help="教師正本（CR canon）")
    ap.add_argument("--serve-sims", type=int, default=160, help="検査点の decide（本番忠実）")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    all_rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.matchup, args.label_sims,
                                                args.serve_sims)) as pool:
        for rows in pool.imap_unordered(play_one, [args.seed_base + i
                                                   for i in range(args.games)]):
            for r in rows:
                all_rows.append(r)
                print(f"  {r['seed']}@T{r['turn']} don={r['don']}/cost{r['min_cost']}"
                      f" 選択={r['choice']} 越境={r['crossed']}"
                      + (f" EV消費={r.get('ev_spend')} EV温存={r.get('ev_end')}"
                         f" gap={r.get('gap')} 欠陥={r.get('defect')}"
                         if "gap" in r else ""), flush=True)
            print(f"  … {len(all_rows)}点 {time.time()-t0:.0f}s", flush=True)

    n = len(all_rows)
    crossed = [r for r in all_rows if r["crossed"]]
    labeled = [r for r in crossed if "gap" in r]
    defects = [r for r in labeled if r["defect"]]
    print(f"\n=== 温存診断（{args.matchup}・{n}点） ===")
    print(f"  境界帯の判断: {n}点  うち越境消費 {len(crossed)}"
          f"（温存/非越境 {n - len(crossed)}）")
    if labeled:
        gaps = np.array([r["gap"] for r in labeled])
        print(f"  越境消費のラベル済み {len(labeled)}点: 欠陥 {len(defects)}点"
              f"（gap≥1/3）・gap 平均 {gaps.mean():+.3f} / 最大 {gaps.max():+.3f}")
    print("DON_RESERVE_PROBE " + json.dumps(
        {"matchup": args.matchup, "n": n, "crossed": len(crossed),
         "labeled": len(labeled), "defects": len(defects), "rows": all_rows},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
