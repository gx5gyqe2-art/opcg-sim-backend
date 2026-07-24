"""乖離診断プローブ（v12）: 候補 vs 既定(gen6) の記録対戦から「選択が分かれた決定点」を
採掘し、レフェリー（gen5 錨・プラン列挙＋CRN 対照評価）で裁く読み取り専用の計器。

背景: v11/v12 でコーチゲート PASS の候補が2連続でアリーナ敗退（19/48・wr 0.396）＝
「7マークのプロファイルは向上するが実戦で gen6 に負ける」構造乖離。問うべきは
**候補は実戦のどこで gen6 より悪い手を打つのか**であり、本プローブがその悪癖カタログを作る。

  1. 記録対戦: run_game（席= make_seat(kind='learned', engine=…)・decide の rng 隔離・
     _RecordObserver）で candidate vs 既定 をペア（同 seed/デッキ・席入替）生成。
  2. 採掘: 候補が**負けた**対局の候補手番（_WINDOWS）を真盤面再生（state_at_action）し、
     候補と既定の decide を同条件比較。equiv キーが異なる点＝乖離点。
  3. 裁定: 乖離点でプラン自動列挙＋同価値バンド判定（referee_labeler と同一機構・教師ネットは
     gen5 固定錨）→ 候補劣（cand の手だけバンド外）/既定劣/両可/両外 に分類。

出力: 乖離点ごとの行＋ `DIVERGE_RESULT`（集計＋行動種の遷移カタログ）。教師は書かない。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/divergence_probe.py \
    --candidate /tmp/cand_value.npz,/tmp/cand_policy.npz --pairs 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random as _random
import time
from collections import Counter

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import replay_runner as RR
import p3_loop as P
import rl_net as RN
import rl_encoder as E
import heldout_decks as HD
from game_driver import make_seat, run_game
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from az_policy import state_context
from opcg_action import legal_action_matrix
from pd_batch_common import pack_policy
from referee_labeler import plan_teacher_visit
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WINDOWS = ("MAIN_ACTION", "SELECT_COUNTER", "SELECT_BLOCKER")
ARGS = None


def _rng_isolated(seat):
    """decide が global random を消費すると scripted 再生（state_at_action）で乱数列が
    ズレるため、record_selfplay_descriptor と同じ隔離を掛ける。"""
    def s(ctx):
        st = _random.getstate()
        try:
            return seat(ctx)
        finally:
            _random.setstate(st)
    return s


def record_vs(db, seed, deckb, cand_eng, best_eng, cand_pid, sims):
    """candidate（cand_pid 席）vs 既定 の記録対戦1局。(descriptor, winner名) を返す。"""
    built = deckb(db, seed)
    l1, c1, l2, c2 = built
    rec = RR._RecordObserver()
    engs = {"p1": cand_eng if cand_pid == "p1" else best_eng,
            "p2": cand_eng if cand_pid == "p2" else best_eng}
    seats = {pid: _rng_isolated(make_seat(kind="learned", sims=sims, engine=engs[pid]))
             for pid in ("p1", "p2")}
    res = run_game(seed, db, seats=seats, deck_builder=lambda *_: built, observers=[rec],
                   legal_moves="skip", invariants="raise", first_player="random")
    desc = {"seed": seed, "first_player": None, "first_player_mode": "random",
            "cpu_player_id": None, "difficulty": "learned",
            "leaders": {"p1": l1.master.card_id if l1 else None,
                        "p2": l2.master.card_id if l2 else None},
            "decks": {"p1": [ci.master.card_id for ci in c1],
                      "p2": [ci.master.card_id for ci in c2]},
            "actions": rec.actions}
    return desc, res.winner


def band_top_keys(game_root, game_serve, vf, pf, m0, name):
    """決定点のプラン列挙＋CRN 評価 → (同価値バンド上位プランの初手 equiv キー集合(repr), entries)。
    referee_labeler.label_decision と同一機構（捲りエスカレーション含む）。プラン不足は (None, None)。"""
    plans = CR.enumerate_turn_plans(game_root, vf, m0, name, max_len=ARGS.plan_len,
                                    beam=ARGS.beam, max_plans=ARGS.max_plans,
                                    log=lambda *a, **k: None)
    if len(plans) <= 1:
        return None, None
    entries = [{"label": "", "keys": keys} for keys, _descs in plans]
    CR._eval_entries(entries, game_root, game_serve, vf, pf, m0, name, ARGS.worlds)
    entries.sort(key=lambda e: (-e["wins"], -e["lifem"]))
    if ARGS.comeback > 0 and entries[0]["wins"] <= 1:
        sub = entries[:min(6, len(entries))]
        CR._eval_entries(sub, game_root, game_serve, vf, pf, m0, name, ARGS.worlds * 4,
                         opp_temp=ARGS.comeback)
        sub.sort(key=lambda e: (-e["wins"], -e["lifem"]))
        entries = sub
    best = entries[0]
    tops = {repr(e["keys"][0]) for e in entries
            if e is best or CR.same_value(best, e, ARGS.band)}
    return tops, entries


def probe_game(db, game_root, game_serve, vf, pf, cand_eng, best_eng, desc, cand_name,
               deadline, sink=None, vocab=None, ev_rec=None):
    """候補敗北局1つを走査: 候補手番の乖離点を裁定し行リストを返す。
    sink 指定時は裁定済み乖離点を教師化して sink へ追加（kind='diverge'・v12 教師化）。"""
    rows = []
    idxs = [i for i, a in enumerate(desc["actions"]) if a.get("player") == cand_name]
    if len(idxs) > ARGS.scan_cap:            # 走査は等間隔サンプル（O(n^2) 再生の抑制）
        stride = len(idxs) / float(ARGS.scan_cap)
        idxs = [idxs[int(k * stride)] for k in range(ARGS.scan_cap)]
    judged = 0
    for i in idxs:
        if judged >= ARGS.max_per_game or time.time() > deadline:
            break
        m0, who = RR.state_at_action(db, desc, i)
        if m0 is None or who != cand_name:
            continue
        pend = m0.get_pending_request(with_request_id=False) or {}
        if pend.get("action") not in _WINDOWS:
            continue
        actor = m0.p1 if m0.p1.name == cand_name else m0.p2
        mv_c = mv_b = None
        try:
            cand_eng._world_seeds = {}
            mv_c = cand_eng.decide(m0, actor, sims=ARGS.sims_cmp,
                                   rng=np.random.default_rng(9500 + 7 * i))
            best_eng._world_seeds = {}
            mv_b = best_eng.decide(m0, actor, sims=ARGS.sims_cmp,
                                   rng=np.random.default_rng(9500 + 7 * i))
        except Exception:
            continue
        if mv_c is None or mv_b is None:
            continue
        try:
            kc = cpu_ai._move_equiv_key(m0, mv_c)
            kb = cpu_ai._move_equiv_key(m0, mv_b)
        except Exception:
            continue
        if repr(kc) == repr(kb):
            continue                                   # 同選択＝乖離なし
        tops, entries = band_top_keys(game_root, game_serve, vf, pf, m0, cand_name)
        if tops is None:
            continue
        judged += 1
        cin, bin_ = repr(kc) in tops, repr(kb) in tops
        verdict = ("cand_bad" if (not cin and bin_) else
                   "best_bad" if (cin and not bin_) else
                   "both_ok" if (cin and bin_) else "both_out")
        tc = int(getattr(m0, "turn_count", 0))
        phase = "early" if tc <= 4 else ("mid" if tc <= 8 else "late")
        rows.append({"idx": i, "turn": tc, "phase": phase, "pend": pend.get("action"),
                     "cand": [mv_c.get("action_type"), kc[1]],
                     "best": [mv_b.get("action_type"), kb[1]], "verdict": verdict})
        print(f"    @{i} T{tc} {pend.get('action')}: cand={mv_c.get('action_type')}:{kc[1]}"
              f" vs gen6={mv_b.get('action_type')}:{kb[1]} → {verdict}", flush=True)
        if sink is not None:
            # v12 教師化: 裁定済み乖離点＝候補の失敗分布そのものから採った反例。バンド判定を
            # そのまま policy 教師（band-top 初手 multi-hot）と value 教師（best 勝率 z）にする。
            legal = game_root.legal_actions(m0)
            legal_keys = []
            for mv in legal:
                try:
                    legal_keys.append(cpu_ai._move_equiv_key(m0, mv))
                except Exception:
                    legal_keys.append(None)
            visit = plan_teacher_visit(legal_keys, entries, ARGS.band)
            if visit is not None:
                best = entries[0]
                z = 2.0 * (best["wins"] / max(best["ok"], 1)) - 1.0
                enc = E.encode(m0, cand_name, vocab, version=ev_rec)
                ctx = state_context(m0, cand_name, vocab, version=ev_rec)
                am = legal_action_matrix(m0, legal, cand_name)
                sink.append((enc, z, (ctx, am, visit)))
    return rows


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="value.npz,policy.npz")
    ap.add_argument("--pairs", type=int, default=4, help="seed ペア数（×2局・席入替）")
    ap.add_argument("--seed0", type=int, default=77000)
    ap.add_argument("--sims-play", type=int, default=120, help="対戦生成の decide sims（arena 同等）")
    ap.add_argument("--sims-cmp", type=int, default=80, help="乖離判定の decide sims")
    ap.add_argument("--sims", type=int, default=32, help="レフェリーのロールアウト sims")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--comeback", type=float, default=0.7)
    ap.add_argument("--plan-len", type=int, default=4)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--max-plans", type=int, default=12)
    ap.add_argument("--scan-cap", type=int, default=24, help="1局あたり走査する候補手番の上限")
    ap.add_argument("--max-per-game", type=int, default=6, help="1局あたり裁定する乖離点の上限")
    ap.add_argument("--max-probe-s", type=float, default=2400.0, help="全体 wall-clock 予算（安全弁）")
    ap.add_argument("--out", default=None,
                    help="裁定済み乖離点を教師バッチ（batch.npz/meta.json・kind='diverge'）として保存")
    ARGS = ap.parse_args()
    CR.ARGS = ARGS

    db = _load_db()
    parts = ARGS.candidate.split(",")
    cand_eng = LearnedEngine(value_path=parts[0],
                             policy_path=parts[1] if len(parts) > 1 else None)
    best_eng = LearnedEngine()                        # 出荷既定（現 gen6）
    # レフェリー（裁定）は gen5 固定錨＝教師ラベルと同じ基準（referee_labeler と同一）。
    vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
    from az_policy import PolicyScorer
    pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    from opcg_sim.src.core.cpu_learned import _net_enc_version
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    vf = P.value_fn_of(vnet, vocab, ev)
    pf = P.priors_fn_of(pnet, vocab, ev)
    game_root = OPCGGame(prune_futile=False)
    game_serve = OPCGGame()
    ids = HD.deck_ids()

    def _deckb(_db, seed):
        l1, c1 = HD.build(_db, ids[seed % len(ids)], "p1")
        l2, c2 = HD.build(_db, ids[(seed + 1) % len(ids)], "p2")
        return l1, c1, l2, c2

    ev_rec = max(E.known_versions())     # 教師の記録は最新符号化版（referee_labeler と同じ）
    sink = [] if ARGS.out else None
    deadline = time.time() + ARGS.max_probe_s
    all_rows = []
    wins = losses = 0
    for p in range(ARGS.pairs):
        seed = ARGS.seed0 + p
        for cand_pid in ("p1", "p2"):
            if time.time() > deadline:
                print("  [予算] プローブ予算超過 → 残り対局をスキップ", flush=True)
                break
            t0 = time.time()
            desc, winner = record_vs(db, seed, _deckb, cand_eng, best_eng, cand_pid,
                                     ARGS.sims_play)
            lost = (winner is not None and winner != cand_pid)
            wins += 0 if lost else (1 if winner == cand_pid else 0)
            losses += 1 if lost else 0
            print(f"pair{p} cand={cand_pid}: winner={winner} {len(desc['actions'])}手 "
                  f"({time.time() - t0:.0f}s){'  → 乖離採掘' if lost else ''}", flush=True)
            if lost:
                all_rows += probe_game(db, game_root, game_serve, vf, pf, cand_eng,
                                       best_eng, desc, cand_pid, deadline,
                                       sink=sink, vocab=vocab, ev_rec=ev_rec)

    if ARGS.out and sink:
        os.makedirs(ARGS.out, exist_ok=True)
        arrays = {"scalars": np.stack([s[0]["scalars"] for s in sink]),
                  "field": np.stack([s[0]["field"] for s in sink]),
                  "card_idx": np.stack([s[0]["card_idx"] for s in sink]),
                  "value": np.array([s[1] for s in sink], dtype=np.float32),
                  "q_root": np.array([s[1] for s in sink], dtype=np.float32),
                  "turns_left": np.full(len(sink), np.nan, dtype=np.float32),
                  "kind": np.array(["diverge"] * len(sink))}
        arrays.update(pack_policy([s[2] for s in sink]))
        np.savez_compressed(os.path.join(ARGS.out, "batch.npz"), **arrays)
        with open(os.path.join(ARGS.out, "meta.json"), "w") as f:
            json.dump({"worker": "dvg", "source": "divergence_probe", "states": len(sink),
                       "schema_version": 2, "games": ARGS.pairs * 2, "worlds": ARGS.worlds,
                       "comeback": ARGS.comeback}, f, ensure_ascii=False)
        print(f"saved: {ARGS.out}/batch.npz ({len(sink)} 乖離教師)", flush=True)

    cnt = Counter(r["verdict"] for r in all_rows)
    trans = Counter(f"{r['cand'][0]}→{r['best'][0]}" for r in all_rows
                    if r["verdict"] == "cand_bad")
    phase = Counter(r["phase"] for r in all_rows if r["verdict"] == "cand_bad")
    print(f"\nDIVERGE_RESULT {json.dumps({'games': ARGS.pairs * 2, 'cand_wins': wins, 'cand_losses': losses, 'judged': len(all_rows), 'verdicts': dict(cnt), 'cand_bad_transitions': dict(trans), 'cand_bad_phase': dict(phase)}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
