"""コーチングスイープ（v8 柱C・docs/cpu_v8_plan.md §3）: 録画1局の決定点をまとめて採点する。

各決定点（真盤面・MAIN_ACTION のみ）で:
  1. ターンプランを自動列挙（counterfactual_referee.enumerate_turn_plans・柱A）
  2. 実際に打たれたプラン（記録アクション列から復元）を同じ CRN 世界線で判定
  3. 最良プランとの差が同価値バンド（柱B）を超えた決定だけを「損失」として報告

教師CPUの答え合わせは**少数局面を深く**が原則（全対局×全決定の常時実行は想定しない）。
コスト制御: --range で決定点を絞る・worlds/sims は既定を軽くしてある。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/coach_sweep.py \
    --tag g3 --player p2 --range 60:96 --worlds 4 --sims 32
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import mark_gate as MG
import replay_reeval as RE
import replay_runner as RR
import p3_loop as P
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def actual_plan_keys(game_root, m0, name, actions, start, fmap):
    """記録アクション列から「実際に打たれたプラン」を equiv キー列として復元する。

    真盤面 m0 に記録手を順に適用し（`resolve_api_action`＝対話/対象欠落も写像）、
    手番が自分から離れるまで（攻撃宣言/TURN_END）を1プランとする。列挙側と同じ終端規約。
    復元不能（ルール分岐）なら None。"""
    acts_idx = [dict(a, _idx=n) for n, a in enumerate(actions)]
    m = m0
    keys, descs = [], []
    i = start
    while i < len(actions):
        rec = acts_idx[i]
        if rec.get("player") != name:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        mv = RR.resolve_api_action(m, actor, rec, frames=fmap, actions=acts_idx)
        if mv is None:
            return None, None
        try:
            keys.append(cpu_ai._move_equiv_key(m, mv))
            descs.append(cpu_ai._describe_move(m, mv) or {})
        except Exception:
            return None, None
        m = game_root.apply(m, mv, name)
        if m is None:
            return None, None
        if game_root.current_player(m) != name:
            break
        i += 1
    return keys, descs


def sweep(db, game_root, game_serve, vf, pf, tag, indices, worlds, band, log=print):
    rec, fbi, actions = CR.GAMES[tag]
    losses = []
    judged = 0
    for i in indices:
        m0, who = RR.state_at_action(db, rec, i, frames=fbi)
        if m0 is None:
            log(f"@{i}: 再生不能（スキップ）: {str(who)[:100]}")
            continue
        pending = m0.get_pending_request()
        if not pending or pending.get("action") != "MAIN_ACTION":
            continue   # プラン比較は自ターンの主導権局面のみ（防御応答・対話は対象外）
        name = who
        akeys, adescs = actual_plan_keys(game_root, m0, name, actions, i, fbi)
        if akeys is None:
            log(f"@{i}: 実プラン復元不能（スキップ）")
            continue
        t0 = time.time()
        auto = CR.enumerate_turn_plans(game_root, vf, m0, name, max_len=ARGS.plan_len,
                                       beam=ARGS.beam, max_plans=ARGS.max_plans, log=log)
        sig = tuple(map(repr, akeys))
        if not any(tuple(map(repr, k)) == sig for k, _d in auto):
            auto.append((akeys, adescs))   # 実プランが列挙から漏れていたら必ず加える
        entries = [{"label": ">".join(CR._step_label(d) for d in descs), "keys": keys,
                    "actual": tuple(map(repr, keys)) == sig,
                    "wins": 0.0, "life": 0.0, "ok": 0, "outcomes": {}}
                   for keys, descs in auto]
        for w in range(worlds):
            world = game_serve.determinize(m0, name, np.random.default_rng(90000 + w * 97))
            for n_e, e in enumerate(entries):
                m = world
                ok = True
                for k in e["keys"]:
                    mv = CR._match_move_by_key(m, game_root.legal_actions(m), k)
                    if mv is None:
                        ok = False; break
                    m = game_serve.apply(m, mv, name)
                    if m is None:
                        ok = False; break
                if not ok:
                    continue
                winner, ld, _et = CR.rollout(game_serve, vf, pf, m, name,
                                             world_seed=90000 + w * 97, rng_seed=w * 7919 + n_e)
                e["outcomes"][w] = (winner == name)
                if winner == name:
                    e["wins"] += 1
                e["life"] += ld
                e["ok"] += 1
        for e in entries:
            e["lifem"] = e["life"] / max(e["ok"], 1)
        entries.sort(key=lambda e: (-e["wins"], -e["lifem"]))
        best = entries[0]
        act = next((e for e in entries if e["actual"]), None)
        judged += 1
        if act is None or act["ok"] == 0:
            log(f"@{i}: 実プランが全世界で不成立（判定不能）")
            continue
        dw = best["wins"] - act["wins"]
        dl = best["lifem"] - act["lifem"]
        # 同価値バンド v2（対判定）: 世界別勝敗の正味不一致 < 3 かつ ライフ差 < band は同価値。
        tie = CR.same_value(best, act, band)
        verdict = "OK（最良）" if act is best else ("同価値" if tie else "損失")
        log(f"@{i} T{m0.turn_count} {verdict}  実際: {act['label']} "
            f"{act['wins']:.0f}/{worlds} L{act['lifem']:+.2f}  ({time.time()-t0:.0f}s)")
        if not tie and act is not best:
            log(f"      最良: {best['label']} {best['wins']:.0f}/{worlds} L{best['lifem']:+.2f}"
                f"  （勝{dw:+.0f}・ライフ{dl:+.2f}）")
            losses.append({"i": i, "turn": m0.turn_count, "actual": act["label"],
                           "best": best["label"], "dw": dw, "dl": dl})
    log(f"\nCOACH_RESULT {tag}: 判定 {judged} 決定・損失 {len(losses)} 件")
    for r in losses:
        log(f"  @{r['i']} T{r['turn']}: {r['actual']} → {r['best']}"
            f"（勝{r['dw']:+.0f}・ライフ{r['dl']:+.2f}）")
    return losses


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g3", help="mark_gate.REPLAYS の録画タグ")
    ap.add_argument("--range", default=None, help="決定 index 範囲 'a:b'（省略=全域）")
    ap.add_argument("--player", default=None,
                    help="採点する側の席（省略=決定点の手番のまま両側）")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--plan-len", type=int, default=4)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--max-plans", type=int, default=16)
    ap.add_argument("--net", default=None, help="value.npz[,policy.npz]（既定=出荷 gen5）")
    ARGS = ap.parse_args()
    CR.ARGS = ARGS   # enumerate_turn_plans / rollout が参照する（sims/plan-len 等）

    db = _load_db()
    if ARGS.net:
        parts = ARGS.net.split(",")
        vnet = RN.ValueNet.load(parts[0])
        pnet = PolicyScorer.load(parts[1]) if len(parts) > 1 else None
    else:
        vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
        pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    vf = P.value_fn_of(vnet, vocab, ev)
    pf = P.priors_fn_of(pnet, vocab, ev)
    game_root = OPCGGame(prune_futile=False)
    game_serve = OPCGGame()

    raw = RE.load_replay_json(MG.REPLAYS[ARGS.tag]); rec = raw.get("replay", raw)
    CR.GAMES = {ARGS.tag: (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                           rec["actions"])}
    actions = rec["actions"]
    lo, hi = 0, len(actions)
    if ARGS.range:
        a, _, b = ARGS.range.partition(":")
        lo, hi = int(a or 0), int(b or len(actions))
    indices = [i for i in range(lo, min(hi, len(actions)))
               if not ARGS.player or actions[i].get("player") == ARGS.player]
    sweep(db, game_root, game_serve, vf, pf, ARGS.tag, indices,
          ARGS.worlds, ARGS.band)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
